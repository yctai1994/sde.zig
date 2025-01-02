fn Stepper(comptime KMAX: usize) type {
    const ATOL: comptime_float = 1e-10;
    const RTOL: comptime_float = 1e-10;
    const KMIN: comptime_int = @trunc(-@log10(@max(1e-12, RTOL)) * 0.6 + 0.5);
    if (KMAX < KMIN + 2) @compileLog("KMAX should be at least ", KMIN, ".\n");

    const IMAX: usize = KMAX + 1;
    _ = IMAX;

    const slice_al: comptime_int = @alignOf([]f64);
    const child_al: comptime_int = @alignOf(f64);
    const slice_sz: comptime_int = @sizeOf(usize) * 2;
    const child_sz: comptime_int = @sizeOf(f64);

    const STEPFAC1: comptime_float = 0.65;
    const STEPFAC2: comptime_float = 0.94;
    const STEPFAC3: comptime_float = 0.02;
    const STEPFAC4: comptime_float = 4.00;

    return struct {
        buffs: [3][]f64, // [dy, ym, yn]
        coeff: [][]f64,
        table: [][]f64, // (nrow = KMAX, ncol = nvar)
        costs: []f64, // KMAX
        hopts: []f64, // KMAX
        works: []f64, // KMAX
        deriv: *const fn (x: f64, src: []f64, des: []f64) void,
        alloc: []u8,

        const Self = @This();

        fn init(allocator: mem.Allocator, deriv: *const fn (x: f64, src: []f64, des: []f64) void, nvar: usize) !*Self {
            const self: *Self = try allocator.create(Self);
            errdefer allocator.destroy(self);

            // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

            const basis_sz: usize = child_sz * nvar;
            const coeff_sz: usize = comptime ((KMAX * KMAX + KMAX) >> 1) * child_sz + KMAX * slice_sz;
            const table_sz: usize = KMAX * basis_sz + KMAX * slice_sz;
            const alloc_sz: usize =
                basis_sz * 3 + // dydx, ym, yn
                coeff_sz + table_sz +
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

        fn step(self: *const Self, dest: []f64, y0: []f64, x0: f64, h0: f64) void {
            const coeff: [][]f64 = self.coeff;
            const table: [][]f64 = self.table;
            const costs: []f64 = self.costs;
            const hopts: []f64 = self.hopts;
            const works: []f64 = self.works;

            var reject: bool = true;
            var k_targ: usize = KMIN;
            var htry: f64 = h0;
            var knew: usize = undefined;
            var hnew: f64 = undefined;
            var k: usize = 0;

            while (reject) {
                reject = false;
                debug.print("[[[ htry = {d} ]]]\n", .{htry});

                k = 0;
                self.mmid(table[k], y0, x0, htry / coeff[0][k], (k + 1) << 1);

                k += 1;
                inner: while (k <= k_targ + 1) : (k += 1) {
                    self.mmid(table[k], y0, x0, htry / coeff[0][k], (k + 1) << 1);
                    self.extp(k);

                    const err: f64 = self.yerr(y0);
                    const expo: f64 = inv(@as(f64, @floatFromInt(2 * k + 1)));
                    const facmin: f64 = pow(STEPFAC3, expo);

                    if (err != 0.0) {
                        const tmp: f64 = STEPFAC2 * pow(STEPFAC1 / err, expo);
                        const fac: f64 = @max(facmin / STEPFAC4, @min(inv(facmin), tmp));
                        hopts[k] = @abs(htry * fac);
                    } else hopts[k] = @abs(htry * inv(facmin));

                    works[k] = costs[k] / hopts[k];

                    if (k < k_targ - 1) continue :inner;

                    if (k == k_targ - 1) {
                        debug.print("(k = {d}) = (k_targ - 1 = {d}) ==> checking...\n", .{ k, k_targ - 1 });
                        if (err <= 1.0) { // err <= 1.0 for k == k_targ - 1
                            // Eq. (17.3.14)
                            knew = if (works[k_targ - 2] < 0.8 * works[k_targ - 1])
                                k_targ - 2
                            else if (works[k_targ - 1] < 0.9 * works[k_targ - 2])
                                @min(k_targ, KMAX - 1)
                            else
                                k_targ - 1;

                            // Eq. (17.3.15)
                            hnew = if (knew == k_targ - 1 or knew == k_targ - 2)
                                hopts[knew]
                            else if (knew == k_targ)
                                hopts[k_targ - 1] * (costs[k_targ] / costs[k_targ - 1])
                            else
                                unreachable;

                            debug.print("  converge with err = {d}!\n", .{err});
                            break :inner; // accept T[k-1, k-1]
                        } else { // err > 1.0 for k == k_targ - 1
                            // Check if the routine cannot achieve convergence @ k_targ + 1.
                            // Fail to converge means err > 1.0 @ k_targ + 1.
                            if (err > sqr(coeff[0][k_targ] * coeff[0][k_targ + 1] / sqr(coeff[0][0]))) { // Eq. (17.3.17)
                                // Eq. (17.3.14)
                                knew = if (works[k_targ - 2] < 0.8 * works[k_targ - 1])
                                    k_targ - 2
                                else if (works[k_targ - 1] < 0.9 * works[k_targ - 2])
                                    @min(k_targ, KMAX - 1)
                                else
                                    k_targ - 1;

                                // Eq. (17.3.15)
                                hnew = if (knew == k_targ - 1 or knew == k_targ - 2)
                                    hopts[knew]
                                else if (knew == k_targ)
                                    hopts[k_targ - 1] * (costs[k_targ] / costs[k_targ - 1])
                                else
                                    unreachable;

                                debug.print("  rejected by err = {d}!\n", .{err});
                                reject = true;
                                break :inner; // reject this htry
                            }
                        }
                    } else if (k == k_targ) {
                        debug.print("(k = {d}) = (k_targ = {d}) ==> checking...\n", .{ k, k_targ });
                        if (err <= 1.0) { // err <= 1.0 for k == k_targ
                            // Eq. (17.3.18)
                            knew = if (works[k_targ - 1] < 0.8 * works[k_targ])
                                k_targ - 1
                            else if (works[k_targ] < 0.9 * works[k_targ - 1])
                                @min(k_targ + 1, KMAX - 1)
                            else
                                k_targ;

                            // Eq. (17.3.19)
                            hnew = if (knew == k_targ - 1 or knew == k_targ)
                                hopts[knew]
                            else if (knew == k_targ + 1)
                                hopts[k_targ] * (costs[k_targ + 1] / costs[k_targ])
                            else
                                unreachable;

                            debug.print("  converge with err = {d}!\n", .{err});
                            break :inner; // accept T[k, k]
                        } else { // err > 1.0 for k == k_targ
                            // Check if the routine cannot achieve convergence @ k_targ + 1.
                            // Fail to converge means err > 1.0 @ k_targ + 1.
                            if (err > sqr(coeff[0][k_targ + 1] / coeff[0][0])) { // Eq. (17.3.20)
                                // Eq. (17.3.18)
                                knew = if (works[k_targ - 1] < 0.8 * works[k_targ])
                                    k_targ - 1
                                else if (works[k_targ] < 0.9 * works[k_targ - 1])
                                    @min(k_targ + 1, KMAX - 1)
                                else
                                    k_targ;

                                // Eq. (17.3.19)
                                hnew = if (knew == k_targ - 1 or knew == k_targ)
                                    hopts[knew]
                                else if (knew == k_targ + 1)
                                    hopts[k_targ] * (costs[k_targ + 1] / costs[k_targ])
                                else
                                    unreachable;

                                debug.print("  rejected by err = {d}!\n", .{err});
                                reject = true;
                                break :inner; // reject this htry
                            }
                        }
                    } else if (k == k_targ + 1) {
                        debug.print("(k = {d}) = (k_targ + 1 = {d}) ==> checking...\n", .{ k, k_targ + 1 });
                        if (err <= 1.0) { // err <= 1.0 for k == k_targ + 1
                            // Eq. (17.3.21)
                            knew = if (works[k_targ - 1] < 0.8 * works[k_targ])
                                k_targ - 1
                            else if (works[k_targ + 1] < 0.9 * works[k_targ])
                                @min(k_targ + 1, KMAX - 1)
                            else
                                k_targ;

                            // Eq. (17.3.19)
                            hnew = if (knew == k_targ - 1 or knew == k_targ)
                                hopts[knew]
                            else if (knew == k_targ + 1)
                                hopts[k_targ] * (costs[k_targ + 1] / costs[k_targ])
                            else
                                unreachable;

                            debug.print("  converge with err = {d}!\n", .{err});
                            break :inner; // accept T[k + 1, k + 1]
                        } else { // err > 1.0 for k == k_targ + 1
                            // Fail to converge, which is err > 1.0 @ k_targ + 1.
                            // Eq. (17.3.18)
                            knew = if (works[k_targ - 1] < 0.8 * works[k_targ])
                                k_targ - 1
                            else if (works[k_targ] < 0.9 * works[k_targ - 1])
                                @min(k_targ + 1, KMAX - 1)
                            else
                                k_targ;

                            // Eq. (17.3.19)
                            hnew = if (knew == k_targ - 1 or knew == k_targ)
                                hopts[knew]
                            else if (knew == k_targ + 1)
                                hopts[k_targ] * (costs[k_targ + 1] / costs[k_targ])
                            else
                                unreachable;

                            debug.print("  rejected by err = {d}!\n", .{err});
                            reject = true;
                            break :inner; // reject this htry
                        }
                    }
                }

                if (reject) {
                    k_targ = @min(k, knew);
                    htry = @min(htry, hnew);
                } else {
                    k_targ = knew;
                    htry = hnew;
                }
            }

            @memcpy(dest, table[0]);
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

        fn extp(self: *const Self, order: usize) void {
            const coeff: [][]f64 = self.coeff;
            const table: [][]f64 = self.table;

            var index: usize = order - 1;
            while (true) : (index -= 1) {
                for (table[index], table[index + 1]) |*p_new, p_old| {
                    p_new.* += coeff[order - index][index] * (p_new.* - p_old);
                }
                if (index == 0) break;
            }
        }

        fn yerr(self: *const Self, y0: []f64) f64 {
            var tmp: f64 = undefined; // scale
            var err: f64 = 0.0;

            for (y0, 0..) |y0_i, i| {
                tmp = ATOL + RTOL * @max(@abs(y0_i), @abs(self.table[0][i]));
                err += sqr((self.table[0][i] - self.table[1][i]) / tmp);
            }
            err = @sqrt(err / @as(f64, @floatFromInt(y0.len)));

            return err;
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
    const KMAX: comptime_int = 10;

    const page = testing.allocator;

    const stepper = try Stepper(KMAX).init(page, gradient, 1);
    defer stepper.deinit(page);

    const x: f64 = -3.0;
    const h: f64 = 0.05;

    var prev: [1]f64 = .{1.0 / 901.0};
    var next: [1]f64 = undefined;

    stepper.step(&next, &prev, x, h);

    debug.print(
        "ans = {d} vs. approx. = {d}\n",
        .{ solution(x + h), next },
    );
}

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

const std = @import("std");
const mem = std.mem;
const debug = std.debug;
const testing = std.testing;
