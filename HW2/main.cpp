#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <functional>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


using namespace std;
using namespace cv;

struct ROIOperation {
    int x, y, size;
    // Define the function type explicitly
    function<void(Mat&, int, int, int, int, int)> operation;
    int param1, param2;

    ROIOperation(int x, int y, int size, function<void(Mat&, int, int, int, int, int)> operation, int param1 = 0, int param2 = 0)
        : x(x), y(y), size(size), operation(operation), param1(param1), param2(param2) {}
};

void convert_pgm(string image_name, string output_name){
    cv::Mat png_image = cv::imread(image_name, cv::IMREAD_GRAYSCALE);

    // Check if the image was loaded successfully
    if (!png_image.empty()) {
        // Save the image in PPM format
        cv::imwrite(output_name, png_image);
        std::cout << "PNG image converted to PGM successfully." << std::endl;
    } else {
        std::cout << "Failed to load the PNG image." << std::endl;
    }
}

int check_value(int value){
    return std::min(255, std::max(0, value));
}

int get_pixel(Mat src,int y,int x){
    int pixel_value = static_cast<int>(src.at<uchar>(y, x));
    // Ensure the pixel value stays within the [0, 255] range
    pixel_value = std::min(255, std::max(0, pixel_value));
    return pixel_value;
}

void set_pixel(Mat src,Mat tgt, int y,int x, int pixel_value){
    tgt.at<uchar>(y, x) = static_cast<uchar>(pixel_value);

}

void add_Grey_ROI(Mat& src, int x, int y, int size, int pixel, int dummy) {
    // Ensure the ROI is within the image bounds
    int roiWidth = min(size, src.cols - x);
    int roiHeight = min(size, src.rows - y);

    for (int i = 0; i < roiHeight; ++i) {
        for (int j = 0; j < roiWidth; ++j) {
            // Get the current pixel value at (x+j, y+i)
            uchar& currentPixel = src.at<uchar>(y + i, x + j);
            // Add the pixel value, ensuring it stays within the 0-255 range
            int newValue = currentPixel + pixel;
            currentPixel = static_cast<uchar>(check_value(newValue));
        }
    }
}

void binarize_Grey_ROI(Mat& src, int x, int y, int size, int threshold, int dummy) {
    int roiWidth = min(size, src.cols - x);
    int roiHeight = min(size, src.rows - y);

    for (int i = 0; i < roiHeight; ++i) {
        for (int j = 0; j < roiWidth; ++j) {
            uchar& currentPixel = src.at<uchar>(y + i, x + j);
            // Binarize the pixel based on the threshold
            currentPixel = (currentPixel > threshold) ? 255 : 0;
        }
    }
}

void scale_Grey(Mat src, Mat tgt, float ratio){
    for (int i=0; i<tgt.rows; ++i)
	{
		for (int j=0; j<tgt.cols; ++j)
		{	
			/* Map the pixel of new image back to original image */
			int i2 = (int)floor((float)i/ratio);
			int j2 = (int)floor((float)j/ratio);
			if (ratio == 2.0) {
                set_pixel(src, tgt, i,j, get_pixel(src,i2,j2));
			}

			if (ratio == 0.5) {
				/* Average the values of four pixels */
                int value = get_pixel(src,i2,j2) + get_pixel(src,i2,j2+1) + get_pixel(src,i2+1,j2) + get_pixel(src,i2+1,j2+1);
                set_pixel(src, tgt, i, j, check_value(value/4));
			}
		}
	}
}

void decrease_brightness_below_T_ROI(Mat& src, int x, int y, int size, int threshold, int value) {
    int roiWidth = min(size, src.cols - x);
    int roiHeight = min(size, src.rows - y);

    for (int i = 0; i < roiHeight; ++i) {
        for (int j = 0; j < roiWidth; ++j) {
            uchar& currentPixel = src.at<uchar>(y + i, x + j);
            if (currentPixel < threshold) {
                // Decrease brightness if below the threshold
                int newValue = currentPixel - value;
                currentPixel = static_cast<uchar>(check_value(newValue));
            }
        }
    }
}

void save_image(Mat tgt, string output_name){
    
    // Save the image in PPM format
    cv::imwrite(output_name, tgt);
    std::cout << "Image " << output_name << " saved successfully" << std::endl;
    if (tgt.empty()) {cout << "Saving an empty image as " << output_name << endl;}
    
}

void flipG(Mat& img) {
    int rows = img.rows, cols = img.cols;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols / 2; ++j) {
            uchar temp = img.at<uchar>(i, j);
            img.at<uchar>(i, j) = img.at<uchar>(i, cols - 1 - j);
            img.at<uchar>(i, cols - 1 - j) = temp;
        }
    }
}

void rotateG90(Mat& img) {
    Mat temp(img.cols, img.rows, img.type());
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            temp.at<uchar>(j, img.rows - 1 - i) = img.at<uchar>(i, j);
        }
    }
    img = temp;
}

void rotateGn90(Mat& img) {
    Mat temp(img.cols, img.rows, img.type());
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            temp.at<uchar>(img.cols - 1 - j, i) = img.at<uchar>(i, j);
        }
    }
    img = temp;
}

void scaleG(Mat& img, double scale) {
    Mat temp = Mat::zeros(img.rows, img.cols, img.type());
    for (int i = 0; i < temp.rows; ++i) {
        for (int j = 0; j < temp.cols; ++j) {
            int srcX = static_cast<int>(j / scale);
            int srcY = static_cast<int>(i / scale);
            if (srcX < img.cols && srcY < img.rows) {
                temp.at<uchar>(i, j) = img.at<uchar>(srcY, srcX);
            }
        }
    }
    img = temp;
}

void brightG(Mat& img, int brightness) {
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            int newPixel = img.at<uchar>(i, j) + brightness;
            img.at<uchar>(i, j) = static_cast<uchar>(check_value(newPixel));
        }
    }
}

void flipC(Mat& img) {
    // Flip image horizontally
    int rows = img.rows, cols = img.cols;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols / 2; ++j) {
            Vec3b temp = img.at<Vec3b>(i, j);
            img.at<Vec3b>(i, j) = img.at<Vec3b>(i, cols - j - 1);
            img.at<Vec3b>(i, cols - j - 1) = temp;
        }
    }
}

void rotateC90(Mat& img) {
    // Rotate image 90 degrees clockwise
    Mat temp = Mat::zeros(img.cols, img.rows, img.type());
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            for (int c = 0; c < 3; ++c) {
                temp.at<Vec3b>(j, img.rows - 1 - i)[c] = img.at<Vec3b>(i, j)[c];
            }
        }
    }
    img = temp;
}

void rotateCn90(Mat& img) {
    // Rotate image -90 degrees (90 degrees counterclockwise)
    Mat temp = Mat::zeros(img.cols, img.rows, img.type());
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            for (int c = 0; c < 3; ++c) {
                temp.at<Vec3b>(img.cols - 1 - j, i)[c] = img.at<Vec3b>(i, j)[c];
            }
        }
    }
    img = temp;
}

void multiplyC(Mat& img, double factor) {
    // Multiplicative color brightness modification
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            Vec3b& pixel = img.at<Vec3b>(i, j);
            for (int c = 0; c < 3; ++c) {
                int newValue = static_cast<int>(pixel[c] * factor);
                pixel[c] = static_cast<uchar>(check_value(newValue));
            }
        }
    }
}

void addC(Mat& img, int brightness) {
    // Additive color brightness modification
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            Vec3b& pixel = img.at<Vec3b>(i, j);
            for (int c = 0; c < 3; ++c) {
                int newValue = pixel[c] + brightness;
                pixel[c] = static_cast<uchar>(check_value(newValue));
            }
        }
    }
}

void histogramStretch(Mat& src, int x, int y, int size, int A, int B) {
    // Use entire image if size is 0
    int startX = (size == 0) ? 0 : x;
    int startY = (size == 0) ? 0 : y;
    int endX = (size == 0) ? src.cols : min(src.cols, x + size);
    int endY = (size == 0) ? src.rows : min(src.rows, y + size);

    // Find the minimum and maximum intensity values
    double Imin = 255, Imax = 0;
    for (int i = startY; i < endY; ++i) {
        for (int j = startX; j < endX; ++j) {
            uchar intensity = src.at<uchar>(i, j);
            if (intensity < Imin) Imin = intensity;
            if (intensity > Imax) Imax = intensity;
        }
    }

    // Define c and d based on Imin and Imax
    double c = 1.05 * Imin;
    double d = 0.95 * Imax;

    // Apply histogram stretching
    for (int i = startY; i < endY; ++i) {
        for (int j = startX; j < endX; ++j) {
            uchar& pixel = src.at<uchar>(i, j);
            if (pixel < c) {
                pixel = A;
            } else if (pixel > d) {
                pixel = B;
            } else {
                pixel = static_cast<uchar>(A + (double)(pixel - c) / (d - c) * (B - A));
            }
        }
    }
}

void localHistogramStretch(Mat& src, int x, int y, int size, int A, int B) {
    // Use entire image if size is 0
    int startX = (size == 0) ? 0 : x;
    int startY = (size == 0) ? 0 : y;
    int roiWidth = (size == 0) ? src.cols : min(src.cols - startX, size);
    int roiHeight = (size == 0) ? src.rows : min(src.rows - startY, size);
    int halfWidth = roiWidth / 2;
    int halfHeight = roiHeight / 2;

    // Apply histogram stretching to each quadrant
    histogramStretch(src, startX, startY, halfHeight, A, B);
    histogramStretch(src, startX + halfWidth, startY, halfHeight, A, B);
    histogramStretch(src, startX, startY + halfHeight, halfHeight, A, B);
    histogramStretch(src, startX + halfWidth, startY + halfHeight, halfHeight, A, B);
}

void generateHistogram(const Mat& src, Mat& histImg, int x, int y, int size) {
    // Initialize histogram array
    int hist[256] = {0};

    // Calculate the ROI
    int roiWidth = min(size, src.cols - x);
    int roiHeight = min(size, src.rows - y);

    // Calculate histogram data
    for (int i = 0; i < roiHeight; ++i) {
        for (int j = 0; j < roiWidth; ++j) {
            uchar intensity = src.at<uchar>(y + i, x + j);
            hist[intensity]++;
        }
    }

    // Create a histogram image (512x512, black background)
    histImg = Mat(512, 512, CV_8UC1, Scalar(0));

    // Normalize histogram to fit in the image height
    int maxHist = *max_element(hist, hist + 256);
    for (int i = 0; i < 256; ++i) {
        hist[i] = (hist[i] * 511) / maxHist;
    }

    // Draw histogram
    for (int i = 0; i < 256; ++i) {
        line(histImg, Point(2 * i, 511), Point(2 * i, 511 - hist[i]), Scalar(255), 1, 8, 0);
    }
}

void rotateG180(Mat& img) {
    // Apply 90-degree rotation twice for 180-degree rotation
    rotateG90(img);
    rotateG90(img);
}

void slidingWindow(Mat& src, int windowSize, int A, int B) {
    // Create a copy of the source image to avoid modifying the original while processing
    Mat srcCopy = src.clone();

    // Iterate over each pixel in the image
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            // Define the local window bounds
            int startX = max(j - windowSize / 2, 0);
            int startY = max(i - windowSize / 2, 0);
            int endX = min(j + windowSize / 2, src.cols - 1);
            int endY = min(i + windowSize / 2, src.rows - 1);

            // Find the minimum and maximum intensity values within the window
            double Imin = 255, Imax = 0;
            for (int wi = startY; wi <= endY; ++wi) {
                for (int wj = startX; wj <= endX; ++wj) {
                    uchar intensity = srcCopy.at<uchar>(wi, wj);
                    if (intensity < Imin) Imin = intensity;
                    if (intensity > Imax) Imax = intensity;
                }
            }

            // Calculate the stretching parameters for the window
            double c = 1.05 * Imin;
            double d = 0.95 * Imax;

            // Stretch the current pixel's intensity
            uchar& pixel = src.at<uchar>(i, j);
            if (pixel < c) {
                pixel = A;
            } else if (pixel > d) {
                pixel = B;
            } else {
                pixel = static_cast<uchar>(A + (double)(pixel - c) / (d - c) * (B - A));
            }
        }
    }
}

void colorHistogramModification(Mat& src, int A, int B, bool modifyR, bool modifyG, bool modifyB) {
    // Split the source image into its R, G, B components
    vector<Mat> channels(3);
    split(src, channels);

    // Apply local histogram stretching to specified channels
    if (modifyR) {
        localHistogramStretch(channels[2], 0, 0, 0, A, B); // Red channel is at index 2
    }
    if (modifyG) {
        localHistogramStretch(channels[1], 0, 0, 0, A, B); // Green channel is at index 1
    }
    if (modifyB) {
        localHistogramStretch(channels[0], 0, 0, 0, A, B); // Blue channel is at index 0
    }

    // Merge the modified channels back into the source image
    merge(channels, src);
}

void rotateC180(Mat& img) {
    // Apply 90-degree rotation twice for 180-degree rotation
    rotateC90(img);
    rotateC90(img);
}

void process_line(const string& line) {
    stringstream ss(line);
    string src_name, tgt_name, function_name;
    vector<ROIOperation> roiOperations;  // Store all ROI operations for this line

    ss >> src_name >> tgt_name;
    Mat src = imread(src_name);  // Load the image in its original color mode
    if (src.empty()) {
        cerr << "Failed to load the image: " << src_name << endl;
        return;
    }

    // Parse the line for operations
    while (ss.good()) {
        ss >> function_name;
        if (function_name.empty()) break;  // No more operations specified


        if (function_name == "hisStretch") {
            // Ensure the image is in grayscale
            if (src.channels() > 1) cvtColor(src, src, COLOR_BGR2GRAY);
            int x = 0, y = 0, size = 0, A, B;
            ss >> x >> y >> size >> A >> B;  // If size is 0, apply to entire image
            Mat histBefore, histAfter;
            generateHistogram(src, histBefore, x, y, (size == 0) ? min(src.cols, src.rows) : size);  // Generate histogram before stretching
            histogramStretch(src, x, y, size, A, B);
            generateHistogram(src, histAfter, x, y, (size == 0) ? min(src.cols, src.rows) : size);  // Generate histogram after stretching

            // Save histogram images
            string histBeforeName = tgt_name + "_hist_before.png";
            string histAfterName = tgt_name + "_hist_after.png";
            imwrite(histBeforeName, histBefore);
            imwrite(histAfterName, histAfter);
        
        } else if (function_name == "localHisStretch") {
            // Ensure the image is in grayscale
            if (src.channels() > 1) cvtColor(src, src, COLOR_BGR2GRAY);

            int x = 0, y = 0, size = 0, A, B;
            ss >> x >> y >> size >> A >> B;  // If size is 0, apply to entire image
            localHistogramStretch(src, x, y, size, A, B);

        } else if (function_name == "rotate_G90") {
            if (src.channels() > 1) cvtColor(src, src, COLOR_BGR2GRAY);
            rotateG90(src);  // Rotate 90 degrees
            save_image(src, tgt_name);
        
        } else if (function_name == "rotate_G180") {
            if (src.channels() > 1) cvtColor(src, src, COLOR_BGR2GRAY);
            rotateG180(src);  // Rotate 180 degrees
            save_image(src, tgt_name);
        
        } else if (function_name == "rotate_G270") {
            if (src.channels() > 1) cvtColor(src, src, COLOR_BGR2GRAY);
            rotateGn90(src);  // Rotate -90 degrees (equivalent to 270 degrees rotation)
            save_image(src, tgt_name);

        } else if (function_name == "slidingWindow") {
            // Ensure the image is in grayscale
            if (src.channels() > 1) cvtColor(src, src, COLOR_BGR2GRAY);

            int windowSize, A, B;
            ss >> windowSize >> A >> B;
            slidingWindow(src, windowSize, A, B);

        } else if (function_name == "hisModification") {
            int modifyR, modifyG, modifyB, A, B;
            ss >> modifyR >> modifyG >> modifyB >> A >> B; // Reading the modification flags for R, G, B and the A, B values
            colorHistogramModification(src, A, B, modifyR, modifyG, modifyB);

        } else if (function_name == "rotate_C90") {
            rotateC90(src);  // Rotate 90 degrees
            save_image(src, tgt_name);
        
        } else if (function_name == "rotate_C180") {
            rotateC180(src);  // Rotate 180 degrees
            save_image(src, tgt_name);
        
        } else if (function_name == "rotate_C270") {
            rotateCn90(src);  // Rotate -90 degrees (equivalent to 270 degrees rotation)
            save_image(src, tgt_name);

        } else if (function_name == "add" || function_name == "binarize" || function_name == "decrease_brightness") {
            // Convert to grayscale if not already
            if (src.channels() > 1) cvtColor(src, src, COLOR_BGR2GRAY);
            int x, y, size, param1 = 0, param2 = 0;  // Initialize parameters to default values
            ss >> x >> y >> size >> param1 >> param2;
            if (function_name == "add") {
                roiOperations.emplace_back(x, y, size, add_Grey_ROI, param1);
            } else if (function_name == "binarize") {
                roiOperations.emplace_back(x, y, size, binarize_Grey_ROI, param1);
            } else if (function_name == "decrease_brightness") {
                roiOperations.emplace_back(x, y, size, decrease_brightness_below_T_ROI, param1, param2);
            }
        
        } else if (function_name == "convert") {
            convert_pgm(src_name, tgt_name);
            return;  // Exit after convert operation
        
        } else if (function_name == "scaleG") {
            // Convert to grayscale if not already
            if (src.channels() > 1) cvtColor(src, src, COLOR_BGR2GRAY);
            double scValue;
            ss >> scValue;
            scaleG(src, scValue);
        
        } else if (function_name == "flipG") {
            // Convert to grayscale if not already
            if (src.channels() > 1) cvtColor(src, src, COLOR_BGR2GRAY);
            flipG(src);
        
        } else if (function_name == "rotateG") {
            // Convert to grayscale if not already
            if (src.channels() > 1) cvtColor(src, src, COLOR_BGR2GRAY);
            int degree;
            ss >> degree;
            if (degree == 90) rotateG90(src);
            else if (degree == -90) rotateGn90(src);
        
        } else if (function_name == "brightG") {
            // Convert to grayscale if not already
            if (src.channels() > 1) cvtColor(src, src, COLOR_BGR2GRAY);
            int brValue;
            ss >> brValue;
            brightG(src, brValue);
        
        } else if (function_name == "roi") {
            int x, y, size;
            ss >> x >> y >> size;
            Rect roi(x, y, size, size);
            src = src(roi).clone();  // Update src to the extracted ROI for subsequent operations
        
        } else if (function_name == "flipC") {
            flipC(src);
        
        } else if (function_name == "rotateC") {
            int degree;
            ss >> degree;
            if (degree == 90) rotateC90(src);
            else if (degree == -90) rotateCn90(src);
    
        } else if (function_name == "multiplyC") {
            double factor;
            ss >> factor;
            multiplyC(src, factor);
        
        } else if (function_name == "addC") {
            int brightness;
            ss >> brightness;
            addC(src, brightness);
        
        } else {
            cerr << "Unsupported function: " << function_name << endl;
        }
    }
    // Apply each ROI operation independently to the source image
    for (const auto& op : roiOperations) {
        op.operation(src, op.x, op.y, op.size, op.param1, op.param2);
    }
    save_image(src, tgt_name);  // Save the final image after processing all operations
    cout << "Image " << tgt_name << " processed and saved successfully." << endl;
}


#define MAXLEN 256
int main() {
    ifstream paramFile("parameters.txt");
    string line;

    if (!paramFile.is_open()) {
        cerr << "Failed to open the parameter file." << endl;
        return 1;
    }

    while (getline(paramFile, line)) {
        process_line(line);
    }

    return 0;
}