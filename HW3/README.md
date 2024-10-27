Image Processing Fundamentals HW-3

Compilation and Execution Steps
-------------------------------
1. Extract the provided zipped folder.
2. Open the project in Visual Studio Code or your preferred IDE.
3. To compile and run the program, open the command palette using `Ctrl+Shift+P`.
4. Search for `Tasks: Run Task` and execute it.
5. Select `C++ build and run main.cpp with parameter file` from the task list. This will compile and execute the program.
6. Among the next provided options, you can just continue without scanning the task output.

Parameter File Format
---------------------
The `parameters.txt` file should contain one or more lines, each specifying a sequence of operations for an input image. The general format is as follows:
inputImage outputImage operation1 [params] operation2 [params] ...

### Example Usage:
baboon.pgm baboon_roi.png roi 50 50 270
baboon_roi.png baboon_roi_his_equal.pgm histequal

This will extract a region of interest (ROI) from `baboon.pgm` and apply histogram equalization to the extracted ROI, saving the result as `baboon_roi_his_equal.pgm`.

Supported Operations
--------------------
Below is a list of supported operations that can be specified in the `parameters.txt` file along with their expected parameters.

### Grayscale Operations:
- `roi X Y SIZE`: Extract an ROI from the image starting at (X, Y) with size SIZE.
- `histequal`: Apply global histogram equalization to the grayscale image.
- `histequalThresh T`: Apply histogram equalization with thresholding to the grayscale image.

### Color Operations:
- `hisEqColor eqR eqG eqB`: Apply histogram equalization to one or more RGB components.
- `convertto_hsv`: Convert RGB image to HSV color space.
- `histequal_V_RGB`: Perform histogram equalization on the V (value) component in RGB space.
- `histequal_H`: Perform histogram equalization on the H (hue) component in HSV space.
- `histequal_S`: Perform histogram equalization on the S (saturation) component in HSV space.

### Advanced Operations:
- `detectQR`: Detect and decode a QR code within the image.
- `rotate_G90/180/270`: Rotate the grayscale image by 90, 180, or 270 degrees.
- `rotate_C90/180/270`: Rotate the color image by 90, 180, or 270 degrees.

For a full list of operations and their parameters, refer to the `process_line` function within the code.