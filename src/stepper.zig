fn Stepper(KMAXX: usize) type {
    if (KMAXX < 2) @compileError("KMAXX should be at least 2.");

    const slice_al: comptime_int = @alignOf([]f64);
    const child_al: comptime_int = @alignOf(f64);
    const slice_sz: comptime_int = @sizeOf(usize) * 2;
    const child_sz: comptime_int = @sizeOf(f64);

    return struct {
        buffs: [3][]f64, // [dydx, ym, yn]
        coeff: [][]f64,
        table: [][]f64, // (nrow = KMAXX, ncol = nvar)
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

            const basis_sz: usize = nvar * child_sz;
            const coeff_sz: usize = comptime ((KMAXX * KMAXX + KMAXX) >> 1) * child_sz + KMAXX * slice_sz;
            const table_sz: usize = KMAXX * basis_sz + KMAXX * slice_sz;
            const alloc_sz: usize = basis_sz * 3 + coeff_sz + table_sz;

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
                    break :inner ptr[0..KMAXX];
                };

                var padding: usize = comptime KMAXX * slice_sz;
                var chunk_sz: usize = comptime KMAXX * child_sz;

                for (0..KMAXX) |k| {
                    temp[k] = inner: {
                        const ptr: [*]align(child_al) f64 = @ptrCast(@alignCast(buff.ptr + padding));
                        break :inner ptr[0 .. KMAXX - k];
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
                    break :inner ptr[0..KMAXX];
                };

                const chunk_sz: usize = nvar * child_sz;
                var padding: usize = KMAXX * slice_sz;

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

            self.deriv = deriv;

            return self;
        }

        fn deinit(self: *const Self, allocator: mem.Allocator) void {
            allocator.free(self.alloc);
            allocator.destroy(self);
        }
    };
}

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

inline fn sqr(x: f64) f64 {
    return x * x;
}

fn gradient(x: f64, y: []f64, dy: []f64) void {
    dy[0] = -200.0 * x * sqr(y[0]);
}

fn solution(x: f64) f64 {
    return 1.0 / (1.0 + 100.0 * sqr(x));
}

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

test "step" {
    const KMAXX: comptime_int = 15;

    const page = testing.allocator;

    const stepper = try Stepper(KMAXX).init(page, gradient, 1);
    defer stepper.deinit(page);

    const x: f64 = -3.0;

    var yprev: [1]f64 = .{1.0 / 901.0};

    {
        const coeff: [][]f64 = stepper.coeff;
        const table: [][]f64 = stepper.table;
        const dydx: []f64 = stepper.buffs[0];
        const ym: []f64 = stepper.buffs[1];
        const yn: []f64 = stepper.buffs[2];

        var nstep: usize = (0 + 1) << 1;
        var hstep: f64 = 0.5 / coeff[0][0];

        try mmid(&yprev, dydx, x, nstep, hstep, table[0], gradient, ym, yn); // integrate to x = -2.5

        for (1..KMAXX) |k| {
            nstep = (k + 1) << 1;
            hstep = 0.5 / coeff[0][k];
            try mmid(&yprev, dydx, x, nstep, hstep, table[k], gradient, ym, yn); // integrate to x = -2.5

            var j: usize = k - 1;
            while (true) : (j -= 1) {
                for (table[j], table[j + 1]) |*p, q| p.* += coeff[k - j][j] * (p.* - q);
                if (j == 0) break;
            }
        }
    }

    debug.print(
        "ans = {d} vs. approx. = {d}\ntable = {d}\n",
        .{ solution(-2.5), stepper.table[0], stepper.table },
    );
}

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

const std = @import("std");
const mem = std.mem;
const debug = std.debug;
const testing = std.testing;

const mmid = @import("./mmid.zig").mmid;
