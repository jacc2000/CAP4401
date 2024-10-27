#include <iostream>
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp> 



using namespace std;
using namespace cv;


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


void add_Grey(Mat src, Mat tgt, int pixel, string tgt_name){
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols  ; ++x) {
            int pixel_value = static_cast<int>(src.at<uchar>(y, x)) + pixel;
            // Ensure the pixel value stays within the [0, 255] range
            pixel_value = std::min(255, std::max(0, pixel_value));
            // tgt.at<uchar>(y, x) = static_cast<uchar>(pixel_value);
            set_pixel(src,tgt,y,x,pixel_value);
        }
    }

} 

void binarize_Grey(Mat src, Mat tgt, int pixel){
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols  ; ++x) {
            int pixel_value = static_cast<int>(src.at<uchar>(y, x));
            // Ensure the pixel value stays within the [0, 255] range
            if(pixel_value > pixel){
                tgt.at<uchar>(y, x) = static_cast<uchar>(255);
            }
            else{
                tgt.at<uchar>(y, x) = static_cast<uchar>(0);
            }
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

// Function to decrease the brightness of pixels in the source image that are below a specified threshold
void decrease_brightness_below_T(Mat src, Mat& tgt, int threshold, int value) {
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            // Get the current pixel value
            int pixel_value = static_cast<int>(src.at<uchar>(y, x));
            if (pixel_value < threshold) {
                pixel_value = check_value(pixel_value - value);
                set_pixel(src, tgt, y, x, pixel_value); // using src as a placeholder
            } else {
                set_pixel(src, tgt, y, x, pixel_value); // using src as a placeholder
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
#define MAXLEN 256
int main(int argc, char** argv)
{
    string src_name = argv[1];
    string tgt_name = argv[2];
    Mat src = imread(src_name, cv::IMREAD_GRAYSCALE);
    Mat src_color = imread(src_name);
    cv::Mat tgt = cv::Mat(src.size(), src.type());
    src.copyTo(tgt);


    char *function_name = argv[3];
    if (strncasecmp(function_name,"convert",MAXLEN)==0) {
        cout << "Converting the image from " << src_name << "to" << tgt_name << endl;
		convert_pgm(src_name, tgt_name);
    }
    else if (strncasecmp(function_name,"addG",MAXLEN)==0) {
        cout << "adding pixel value to gray scale image" << src_name <<endl;
        int pixel_value = atoi(argv[4]);
		add_Grey(src,tgt,pixel_value, tgt_name);
        save_image(tgt,tgt_name);
        
    }
    else if (strncasecmp(function_name,"binarizeG",MAXLEN)==0) {
        cout << "binarizing pixel values for " << src_name <<endl;
        int pixel_value = atoi(argv[4]);
		binarize_Grey(src,tgt,pixel_value);
        save_image(tgt,tgt_name);
        
    }
    else if (strncasecmp(function_name,"scaleG",MAXLEN)==0) {
        cout << "scaling image for " << src_name <<endl;
        float ratio = stof(argv[4]);

        cv::Mat resized_image((int)((float)src.cols * ratio),  (int)((float)src.cols * ratio), src.type());
		scale_Grey(src,resized_image,ratio);
        cout << "size of the target is " << resized_image.size() << endl;
        save_image(resized_image,tgt_name);
        
    }
    // Condition to handle the decrease_brightness function
    else if (strncasecmp(function_name, "decrease_brightness", MAXLEN) == 0) {
        cout << "decreasing brightness for pixels below threshold in " << src_name << endl;
        int threshold = atoi(argv[4]);
        int value = atoi(argv[5]);
        decrease_brightness_below_T(src, tgt, threshold, value);
        save_image(tgt, tgt_name);
    }
    else{
        cout << "No such function" << function_name << endl;
    }
    
    return 0;
    
    
}
