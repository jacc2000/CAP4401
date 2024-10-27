Image Processing Fundamentals HW-2

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
The parameters.txt file should contain one or more lines, each specifying a sequence of operations for an input image. The general format is as follows:
`inputImage outputImage operation1 [params] operation2 [params] ...`

Example:

# For histogram stretching and local histogram stretching
inputImage outputImage hisStretch [params] localHisStretch [params]

# For color histogram modification
inputImage outputImage hisModification [params]

# For rotating images
inputImage outputImage rotate_[G|C][90|180|270]

Where `hisStretch` and `localHisStretch` apply histogram stretching operations, `hisModification` applies color histogram modifications, and `rotate_[G|C][90|180|270]` specifies rotation operations for grayscale (G) or color (C) images by 90, 180, or 270 degrees.

Supported Operations
--------------------

### Histogram Modification Operations:

- **hisStretch X Y SIZE A B**: Apply histogram stretching within ROI defined by top-left corner (X, Y), SIZE, and intensity range [A, B].
- **localHisStretch X Y SIZE A B**: Apply local histogram stretching within ROI, dividing it into four quadrants and applying histogram stretching with intensity range [A, B] to each quadrant.

### Color Processing Operations:

- **hisModification R G B A B**: Apply local histogram modification to R, G, B components with intensity range [A, B]. Set R, G, B to 1 to modify that channel, 0 to leave it unchanged.

### Augmentation of Images:

- **rotate_[G|C][90|180|270]**: Rotate grayscale (G) or color (C) images by 90, 180, or 270 degrees.

Note: For `hisStretch` and `localHisStretch`, if X, Y, and SIZE are all 0, the operation applies to the entire image.

