Image Processing Fundamentals HW-1

Compilation and Execution Steps
1. Extract the provided zipped folder.
2. Open the project in Visual Studio Code or your preferred IDE.
3. To compile and run the program, open the command palette using `Ctrl+Shift+P`.
4. Search for `Tasks: Run Task` and execute it.
5. Select `C++ build and run main.cpp with parameter file` from the task list. This will compile and execute the program.
6. Among the next provided options you can just continue without scanning the task output.

#Parameter File Format
The parameters.txt file should contain one or more lines, each specifying a sequence of operations for an input image. The general format is as follows:
inputImage outputImage operation1 [params] operation2 [params] ...

##Example:
for a.
inputImage outputImage functionName1 rX1 rY1 rS1 functionName2 rX2 rY2 rS2 functionName3 rX3 rY3 rS3
where rX1, rY1, and rS1 are the top left pixels, and the size of ROI 1
           rX2, rY2, and rS2 are the top left pixels, and the size of ROI 2
           rX3, rY3, and rS3 are the top left pixels, and the size of ROI 3
           functionName1, functionName2, and functionName3 are the user-defined functions for the ROI.

for b. and c.
inputImage outputImage flipImage degree
where flipImage is a user-defined function to flip the image

inputImage outputImage brightness brValue
where brightness is a user-defined function to perform the brightness
           brValue is the brightness value between 0 and 50

#Supported Operations

##Grayscale Operations:

add X Y SIZE PIXEL: Add PIXEL value to the grayscale intensity within ROI defined by top-left corner (X, Y) and SIZE.
binarize X Y SIZE THRESHOLD: Binarize pixels within the ROI based on THRESHOLD.
decrease_brightness X Y SIZE THRESHOLD VALUE: Decrease brightness for pixels below THRESHOLD by VALUE within the ROI.
flipG: Horizontally flip the grayscale image.
rotateG DEGREE: Rotate the grayscale image by DEGREE (90 or -90).

##Color Operations:

flipC: Horizontally flip the color image.
rotateC DEGREE: Rotate the color image by DEGREE (90 or -90).
multiplyC FACTOR: Apply multiplicative brightness modification to the color image by FACTOR.
addC BRIGHTNESS: Apply additive brightness modification to the color image by BRIGHTNESS.

##ROI Extraction:

roi X Y SIZE: Extract ROI from the image starting at (X, Y) with size SIZE.


