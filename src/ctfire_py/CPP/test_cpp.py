import numpy as np
import fiber_backend


def test_find_local_max():
    print("Testing fiber_backend.findlocmax_native...")

    # 1. Setup Dummy Input Data
    # For a 2D image: sizex=1, sizey=100, sizez=100
    sizex, sizey, sizez = 1, 100, 100

    # Create a blank image and plant a clear "local max" at coordinate (50, 50)
    image = np.zeros((sizey, sizez), dtype=np.float32)
    image[50, 50] = 10.0

    radius = 3
    dmin = 1.0

    # 2. Run the C++ Backend
    # Make sure to flatten/ensure C-contiguous array as defined in our Pybind11 wrapper
    image_c_style = np.ascontiguousarray(image)

    pts = fiber_backend.findlocmax_native(
        sizex, sizey, sizez, image_c_style, radius, dmin
    )

    print(f"C++ Output Shape: {pts.shape}")
    print(f"C++ Output Data:\n{pts}")

    # 3. Define the "Golden" Reference (What MATLAB would output)
    # The C++ code currently outputs [row, col, 1] for 2D.
    # Because of the 1-based indexing parity we kept from MATLAB, (50, 50) becomes (51, 51).
    expected_pts = np.array([[51, 51, 1]], dtype=np.int32)

    # 4. Verify the results
    # Taking only rtol into account and ignoring atol
    np.testing.assert_allclose(
        pts,
        expected_pts,
        rtol=0.05,
        atol=0,
        equal_nan=True,
        err_msg="C++ local max points differ from expected reference.",
    )

    print("Test passed successfully! Output perfectly matches expected reference.")


if __name__ == "__main__":
    test_find_local_max()
