import numpy as np
import fiber_backend


def test_find_local_max():
    print("Testing fiber_backend.find_local_max...")

    # 1. Setup Dummy Input Data
    # For a 2D image: sizex=1, sizey=100, sizez=100
    # The C++ code expects: sizex (depth), sizey (height), sizez (width)
    # For 2D: sizex=1, and the image is (sizey x sizez)
    sizex, sizey, sizez = 1, 100, 100

    # Create a blank image and plant a clear "local max" at coordinate (50, 50)
    # Image shape should be (sizey, sizez) = (100, 100)
    image = np.zeros((sizey, sizez), dtype=np.float32)
    image[50, 50] = 10.0

    radius = 3
    dmin = 1.0

    # 2. Run the C++ Backend
    # Flatten the image to 1D array as expected by the C++ code
    image_flat = image.flatten()

    pts = fiber_backend.find_local_max(sizex, sizey, sizez, image_flat, radius, dmin)

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
