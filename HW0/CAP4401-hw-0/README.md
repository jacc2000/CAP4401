Image Processing Fundamentals HW-0

Compilation and Execution Steps
1. Extract the provided zipped folder.
2. Open the project in Visual Studio Code or your preferred IDE.
3. To compile and run the program, open the command palette using `Ctrl+Shift+P`.
4. Search for `Tasks: Run Task` and execute it.
5. Select `C++ build and run main.cpp` from the task list. This will compile and execute the program.
6. Among the next provided options you can just continue without scanning the task output.

Implemented Functions
1. **Convert to PGM**: Converts an image to PGM format.
2. **Add Grey**: Increases the intensity of a grayscale image.
3. **Binarize Grey**: Applies basic image thresholding.
4. **Scale Grey**: Scales an image by a specified ratio using pixel averaging (for reduction) and replication (for expansion).
5. **Decrease Brightness Below Threshold**: Decreases the brightness of pixels below a specified threshold.

Parameter File Format: `<inputImage> <outputImage> <functionName> <params...>`

Example:
baboon.pgm baboon_output_stretch.pgm add 50 
baboon.pgm baboon_output_equalize.pgm binarize 50 
baboon.png baboon_output_decreased_100_100.pgm decrease_brightness 100 100
