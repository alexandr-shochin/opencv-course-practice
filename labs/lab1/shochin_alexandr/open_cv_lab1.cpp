#include <opencv2\opencv.hpp>
#include <stdlib.h>

using namespace std;
using namespace cv;

#define MASK_SIZE 3
#define DELAY     0

#define MIN_BLUR_COEF 1.0f
#define MAX_BLUR_COEF 100.0f

#define MIN_LOW_TRESHOLD  1.0f
#define MAX_LOW_TRESHOLD  100.0f
#define MIN_HIGH_TRESHOLD 1.0f

#define FAIL    -1
#define SUCCESS  1
#define NUM_ARGS 5

int Calc(Mat inputImage, float lowTreshold, float highTreshold, float blurCoef)
{
	Mat edges; Canny(inputImage, edges, lowTreshold, highTreshold);
	subtract(Scalar::all(255), edges, edges);
	Mat dstMap; distanceTransform(edges, dstMap, ::CV_DIST_L2, MASK_SIZE);
	Mat _integral; integral(inputImage, _integral, CV_32FC3);
	Mat outputImage = Mat(inputImage.rows, inputImage.cols, CV_8UC3);

	for (int Y = 0; Y < inputImage.rows; Y++) {
		for (int X = 0; X < inputImage.cols; X++) {

			int kernel = (int)(blurCoef * dstMap.at<float>(Point(X, Y)));
			kernel = max(1, kernel);

			int maxX = min(inputImage.cols - 1, X + kernel / 2);
			int maxY = min(inputImage.rows - 1, Y + kernel / 2);
			int minX = max(0, X - kernel / 2);
			int minY = max(0, Y - kernel / 2);

			Vec3d tmp(0.0f, 0.0f, 0.0f);
			tmp =
				_integral.at<Vec3f>(Point(maxX + 1, maxY + 1)) -
				_integral.at<Vec3f>(Point(maxX + 1, minY)) -
				_integral.at<Vec3f>(Point(minX, maxY + 1)) +
				_integral.at<Vec3f>(Point(minX, minY));
			tmp /= (maxX - minX + 1) * (maxY - minY + 1);

			outputImage.at<Vec3b>(Point(X, Y)) = tmp;
		}
	}

	imshow("inputImage", inputImage); imshow("outputImage", outputImage);
	waitKey(DELAY); return SUCCESS;
}

int main(int argc, char** argv)
{
	if (argc != NUM_ARGS) {
		printf("invalid arguments: argc != 5\n"); return FAIL; }

	Mat inputImage = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	if (inputImage.data == NULL) {
		printf("invalid image: image = NULL\n"); return FAIL; }

	float lowTreshold = atof(argv[2]);
	if (lowTreshold < MIN_LOW_TRESHOLD || lowTreshold > MAX_LOW_TRESHOLD) { 
		printf("invalid lowTreshold: lowTreshold < 1 || lowTreshold > 100\n");  return FAIL; }

	float highTreshold = atof(argv[3]);
	if (highTreshold < MIN_HIGH_TRESHOLD) {
		printf("invalid highTreshold: highTreshold < 1\n"); return FAIL; }

	float blurCoef = atof(argv[4]);
	if (blurCoef < MIN_BLUR_COEF || blurCoef > MAX_BLUR_COEF) { 
		printf("invalid blurCoef: blurCoef < 1.0 || blurCoef > 100.0\n"); return FAIL; }

	if (highTreshold < lowTreshold) {
		printf("err: highTreshold < lowTreshold\n"); return FAIL; }

	return Calc(inputImage, lowTreshold, highTreshold, blurCoef);
}
