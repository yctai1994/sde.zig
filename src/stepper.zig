fn Stepper(comptime KMAX: usize) type {
    const ATOL: comptime_float = 1e-9;
    const RTOL: comptime_float = 1e-9;
    const KMIN: comptime_int = @trunc(-@log10(RTOL) * 0.6 + 0.5);
    if (KMAX < KMIN + 2) @compileLog("KMAX should be at least ", KMIN, ".\n");

    const IMAXX: usize = KMAX + 1;
    _ = IMAXX;

    const slice_al: comptime_int = @alignOf([]f64);
    const child_al: comptime_int = @alignOf(f64);
    const slice_sz: comptime_int = @sizeOf(usize) * 2;
    const child_sz: comptime_int = @sizeOf(f64);

    const STEPFAC1: comptime_float = 0.65;
    const STEPFAC2: comptime_float = 0.94;
    const STEPFAC3: comptime_float = 0.02;
    const STEPFAC4: comptime_float = 4.00;

    return struct {
        buffs: [3][]f64, // [dydx, ym, yn]
        coeff: [][]f64,
        table: [][]f64, // (nrow = KMAX, ncol = nvar)
        scale: []f64, // nvar
        costs: []f64, // KMAX
        hopts: []f64, // KMAX
        works: []f64, // KMAX
        deriv: *const fn (x: f64, src: []f64, des: []f64) void,
        alloc: []u8,

        const Self = @This();

        fn init(
            allocator: mem.Allocator,
            deriv: *const fn (x: f64, src: []f64, des: []f64) void,
            nvar: usize,
        ) !*Self {
            const self: *Self = try allocator.create(Self);
            errdefer allocator.destroy(self);

            // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

            const basis_sz: usize = child_sz * nvar;
            const coeff_sz: usize = comptime ((KMAX * KMAX + KMAX) >> 1) * child_sz + KMAX * slice_sz;
            const table_sz: usize = KMAX * basis_sz + KMAX * slice_sz;
            const alloc_sz: usize =
                basis_sz * 3 + // dydx, ym, yn
                coeff_sz + table_sz +
                basis_sz + //  scale
                KMAX * child_sz * 3; // costs, hopts, works

            self.alloc = try allocator.alloc(u8, alloc_sz);

            var addr_lo: usize = 0;
            var addr_hi: usize = 0;

            // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

            for (0..3) |n| {
                addr_hi += basis_sz;

                self.buffs[n] = blk: {
                    const ptr: [*]align(child_al) f64 = @ptrCast(@alignCast(self.alloc[addr_lo..addr_hi].ptr));
                    break :blk ptr[0..nvar];
                };

                addr_lo += basis_sz;
            }

            // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

            self.coeff = outer: {
                addr_hi += coeff_sz;

                const buff: []u8 = self.alloc[addr_lo..addr_hi];
                const temp: [][]f64 = inner: {
                    const ptr: [*]align(slice_al) []f64 = @ptrCast(@alignCast(buff.ptr));
                    break :inner ptr[0..KMAX];
                };

                var padding: usize = comptime KMAX * slice_sz;
                var chunk_sz: usize = comptime KMAX * child_sz;

                for (0..KMAX) |k| {
                    temp[k] = inner: {
                        const ptr: [*]align(child_al) f64 = @ptrCast(@alignCast(buff.ptr + padding));
                        break :inner ptr[0 .. KMAX - k];
                    };
                    padding += chunk_sz;
                    chunk_sz -= child_sz;
                }

                for (temp[0], 0..) |*p, i| p.* = @as(f64, @floatFromInt((i + 1) << 1));
                for (temp[1..], 1..) |row, k| {
                    for (row, 0.., k..) |*p, i, ipk| { // ipk := i + k
                        p.* = 1.0 / (sqr(temp[0][i] / temp[0][ipk]) - 1.0);
                    }
                }

                addr_lo += coeff_sz;
                break :outer temp;
            };

            // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

            self.table = outer: {
                addr_hi += table_sz;

                const buff: []u8 = self.alloc[addr_lo..addr_hi];
                const temp: [][]f64 = inner: {
                    const ptr: [*]align(slice_al) []f64 = @ptrCast(@alignCast(buff.ptr));
                    break :inner ptr[0..KMAX];
                };

                const chunk_sz: usize = nvar * child_sz;
                var padding: usize = KMAX * slice_sz;

                for (temp) |*row| {
                    row.* = inner: {
                        const ptr: [*]align(child_al) f64 = @ptrCast(@alignCast(buff.ptr + padding));
                        break :inner ptr[0..nvar];
                    };
                    padding += chunk_sz;
                }

                addr_lo += table_sz;
                break :outer temp;
            };

            // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

            addr_hi += basis_sz;

            self.scale = blk: {
                const ptr: [*]align(child_al) f64 = @ptrCast(@alignCast(self.alloc[addr_lo..addr_hi].ptr));
                break :blk ptr[0..nvar];
            };

            addr_lo += basis_sz;

            // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

            addr_hi += comptime KMAX * child_sz;

            self.costs = blk: {
                const ptr: [*]align(child_al) f64 = @ptrCast(@alignCast(self.alloc[addr_lo..addr_hi].ptr));
                const tmp: []f64 = ptr[0..KMAX];

                tmp[0] = self.coeff[0][0] + 1.0;
                for (1..KMAX) |i| tmp[i] = tmp[i - 1] + self.coeff[0][i];

                break :blk tmp;
            };

            addr_lo += comptime KMAX * child_sz;

            // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

            addr_hi += comptime KMAX * child_sz;

            self.hopts = blk: {
                const ptr: [*]align(child_al) f64 = @ptrCast(@alignCast(self.alloc[addr_lo..addr_hi].ptr));
                break :blk ptr[0..KMAX];
            };

            addr_lo += comptime KMAX * child_sz;

            // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

            addr_hi += comptime KMAX * child_sz;

            self.works = blk: {
                const ptr: [*]align(child_al) f64 = @ptrCast(@alignCast(self.alloc[addr_lo..addr_hi].ptr));
                break :blk ptr[0..KMAX];
            };

            addr_lo += comptime KMAX * child_sz;

            // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

            self.deriv = deriv;

            return self;
        }

        fn deinit(self: *const Self, allocator: mem.Allocator) void {
            allocator.free(self.alloc);
            allocator.destroy(self);
        }

        fn step(self: *const Self, dest: []f64, y0: []f64, x0: f64, htry: f64) void {
            self.mmid(self.table[0], y0, x0, htry / self.coeff[0][0], (0 + 1) << 1);

            for (1..KMAX) |k| {
                self.mmid(self.table[k], y0, x0, htry / self.coeff[0][k], (k + 1) << 1);

                var j: usize = k - 1;
                while (true) : (j -= 1) {
                    for (self.table[j], self.table[j + 1]) |*p, q| p.* += self.coeff[k - j][j] * (p.* - q);
                    if (j == 0) break;
                }

                var err: f64 = 0.0;
                for (self.scale, 0..) |*scale, i| {
                    scale.* = ATOL + RTOL * @max(@abs(y0[i]), @abs(self.table[0][i]));
                    err += sqr((self.table[0][i] - self.table[1][i]) / scale.*);
                }
                err = @sqrt(err / @as(f64, @floatFromInt(y0.len)));
                debug.print("err = {d}\n", .{err});

                const expo: f64 = inv(@as(f64, @floatFromInt(2 * k + 1)));
                const facmin: f64 = pow(STEPFAC3, expo);

                if (err == 0.0) {
                    const fac: f64 = inv(facmin);
                    self.hopts[k] = @abs(htry * fac);
                } else {
                    const tmp: f64 = STEPFAC2 * pow(STEPFAC1 / err, expo);
                    const fac: f64 = @max(facmin / STEPFAC4, @min(inv(facmin), tmp));
                    self.hopts[k] = @abs(htry * fac);
                }

                self.works[k] = self.costs[k] / self.hopts[k];
            }

            @memcpy(dest, self.table[0]);
        }

        fn mmid(self: *const Self, dest: []f64, y0: []f64, x0: f64, h: f64, order: usize) void {
            const dy: []f64 = self.buffs[0];
            const ym: []f64 = self.buffs[1];
            const yn: []f64 = self.buffs[2];

            // First Step:
            @memcpy(ym, y0);
            for (yn, y0, dy) |*yn_i, y0_i, dy_i| yn_i.* = y0_i + h * dy_i;

            var x: f64 = x0 + h;
            self.deriv(x, yn, dest); // Will use dest for temporary storage of derivatives.
            const h2: f64 = 2.0 * h;

            // General Step
            var swap: f64 = undefined;
            for (1..order) |_| {
                for (ym, yn, dest) |*ym_i, *yn_i, dest_i| {
                    swap = ym_i.* + h2 * dest_i;
                    ym_i.* = yn_i.*;
                    yn_i.* = swap;
                }
                x += h;
                self.deriv(x, yn, dest);
            }

            // Last step.
            for (dest, ym, yn) |*dest_i, ym_i, yn_i| dest_i.* = 0.5 * (ym_i + yn_i + h * dest_i.*);
        }
    };
}

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

inline fn sqr(x: f64) f64 {
    return x * x;
}

inline fn inv(x: f64) f64 {
    return 1.0 / x;
}

inline fn pow(base: f64, expo: f64) f64 {
    return @exp(expo * @log(base));
}

fn gradient(x: f64, y: []f64, dy: []f64) void {
    dy[0] = -200.0 * x * sqr(y[0]);
}

fn solution(x: f64) f64 {
    return 1.0 / (1.0 + 100.0 * sqr(x));
}

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

test "step" {
    const KMAX: comptime_int = 8;

    const page = testing.allocator;

    const stepper = try Stepper(KMAX).init(page, gradient, 1);
    defer stepper.deinit(page);

    const x: f64 = -3.0;
    const h: f64 = 0.01;

    var prev: [1]f64 = .{1.0 / 901.0};
    var next: [1]f64 = undefined;

    stepper.step(&next, &prev, x, h);

    debug.print(
        "ans = {d} vs. approx. = {d}\ntable = {d}\n",
        .{ solution(x + h), next, stepper.table },
    );
    debug.print("stepper.costs = {d}\n", .{stepper.costs});
    debug.print("stepper.hopts = {d}\n", .{stepper.hopts});
    debug.print("stepper.works = {d}\n", .{stepper.works});
}

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

const std = @import("std");
const mem = std.mem;
const debug = std.debug;
const testing = std.testing;
