fn Stepper(KMAXX: usize) type {
    if (KMAXX < 2) @compileError("KMAXX should be at least 2.");

    const slice_al: comptime_int = @alignOf([]f64);
    const child_al: comptime_int = @alignOf(f64);
    const slice_sz: comptime_int = @sizeOf(usize) * 2;
    const child_sz: comptime_int = @sizeOf(f64);
    const coeff_sz: comptime_int = ((KMAXX * KMAXX + KMAXX) >> 1) * child_sz + KMAXX * slice_sz;

    return struct {
        coeff: [][]f64,
        table: []f64,

        const Self = @This();

        fn init(allocator: mem.Allocator) !*Self {
            const self: *Self = try allocator.create(Self);
            errdefer allocator.destroy(self);

            // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

            self.table = try allocator.alloc(f64, KMAXX);
            errdefer allocator.free(self.table);

            // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

            self.coeff = outer: {
                const buff: []u8 = try allocator.alloc(u8, coeff_sz);
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

                break :outer temp;
            };

            errdefer {
                const ptr: [*]u8 = @ptrCast(@alignCast(self.coeff.ptr));
                const len: usize = coeff_sz;
                allocator.free(ptr[0..len]);
            }

            // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

            return self;
        }

        fn deinit(self: *const Self, allocator: mem.Allocator) void {
            {
                const ptr: [*]u8 = @ptrCast(@alignCast(self.coeff.ptr));
                const len: usize = coeff_sz;
                allocator.free(ptr[0..len]);
            }

            allocator.free(self.table);
            allocator.destroy(self);
        }
    };
}

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

fn sqr(x: f64) f64 {
    return x * x;
}

fn gradient(x: f64, y: []f64, dy: []f64) void {
    dy[0] = -200.0 * x * sqr(y[0]);
}

fn solution(x: f64) f64 {
    return 1.0 / (1.0 + 100.0 * sqr(x));
}

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

test "step" {
    const KMAXX: comptime_int = 15;

    const page = testing.allocator;

    const stepper = try Stepper(KMAXX).init(page);
    defer stepper.deinit(page);

    const x: f64 = -3.0;
    var yin: [1]f64 = .{1.0 / 901.0};
    var yout: [1]f64 = undefined;
    var dydx: [1]f64 = undefined;

    // buffers
    var ym: [1]f64 = undefined;
    var yn: [1]f64 = undefined;

    {
        const coeff: [][]f64 = stepper.coeff;
        const table: []f64 = stepper.table;

        try mmid(&yin, &dydx, x, (0 + 1) << 1, 0.5 / coeff[0][0], &yout, gradient, &ym, &yn); // integrate to x = -2.5
        table[0] = yout[0];

        for (1..KMAXX) |n| {
            try mmid(&yin, &dydx, x, (n + 1) << 1, 0.5 / coeff[0][n], &yout, gradient, &ym, &yn); // integrate to x = -2.5
            table[n] = yout[0];
            var k: usize = n - 1;
            while (true) : (k -= 1) {
                table[k] += coeff[n - k][k] * (table[k] - table[k + 1]);
                if (k == 0) break;
            }
        }
    }

    debug.print(
        "ans = {d} vs. approx. = {d}\ntable = {d}\n",
        .{ solution(-2.5), stepper.table[0], stepper.table },
    );
}

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

const std = @import("std");
const mem = std.mem;
const debug = std.debug;
const testing = std.testing;

const mmid = @import("./mmid.zig").mmid;
