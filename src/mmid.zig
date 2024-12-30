/// Modified midpoint step.
/// At `xstart`, input the dependent variable vector `y` and its derivative vector `dydx`.
/// Also input is htot, the total step to be made, and nstep, the number of substeps to be used.
/// The output is returned as `yout`, which need not be a distinct array from `y`;
/// if it is distinct, however, then `y` and `dydx` are returned unmodified.
fn mmid(
    allocator: mem.Allocator,
    y: []f64,
    dydx: []f64,
    xstart: f64,
    htot: f64,
    nstep: usize,
    yout: []f64,
    derivs: *const fn (x: f64, yn: []f64, yout: []f64) void,
) !void {
    debug.assert(y.len == dydx.len);
    debug.assert(y.len == yout.len);

    // Allocations:
    const ym: []f64 = try allocator.alloc(f64, y.len);
    defer allocator.free(ym);

    const yn: []f64 = try allocator.alloc(f64, y.len);
    defer allocator.free(yn);

    // Initialization:
    const h: f64 = htot / nstep; // stepsize of this trip.

    // First Step:
    for (0..y.len) |i| {
        ym[i] = y[i];
        yn[i] = y[i] + h * dydx[i];
    }

    var xnew: f64 = xstart + h;
    derivs(xnew, yn, yout); // Will use yout for temporary storage of derivatives.
    const h2: f64 = 2.0 * h;

    // General Step
    var swap: f64 = undefined;
    for (1..nstep) |_| {
        for (0..y.len) |i| {
            swap = ym[i] + h2 * yout[i];
            ym[i] = yn[i];
            yn[i] = swap;
        }
        xnew += h;
        derivs(xnew, yn, yout);
    }

    // Last step.
    for (0..y.len) |i| {
        yout[i] = 0.5 * (ym[i] + yn[i] + h * yout[i]);
    }
}

const std = @import("std");
const mem = std.mem;
const debug = std.debug;
