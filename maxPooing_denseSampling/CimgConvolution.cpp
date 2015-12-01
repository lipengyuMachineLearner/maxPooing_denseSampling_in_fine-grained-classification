#include "CimgConvolution.h"
#include <fstream>
#include <string>
using namespace std;

void CimgConvolution::save(string fileName)
{
	row_ = data_.rows;
	col_ = data_.cols;

	ofstream outfileBin;
	outfileBin.open(fileName, ios::binary);

	outfileBin.write((char *)&maxValue_, sizeof(maxValue_));
	outfileBin.write((char *)&maxPosition_.x, sizeof(maxPosition_.x));
	outfileBin.write((char *)&maxPosition_.y, sizeof(maxPosition_.y));
	outfileBin.write((char *)&width_, sizeof(width_));
	outfileBin.write((char *)&height_, sizeof(height_));

	outfileBin.write((char *)&row_, sizeof(row_));
	outfileBin.write((char *)&col_, sizeof(col_));
	for (int h = 0; h < row_; h++)
		for (int w = 0; w < col_; w++)
			{
				float tmp = data_.at<float>(h, w);
				outfileBin.write((char *)&tmp, sizeof(tmp));
			}

	outfileBin.close();
}

void CimgConvolution::load(string fileName)
{
	ifstream infileBin;
	infileBin.open(fileName, ios::binary);

	infileBin.read((char *)&maxValue_, sizeof(maxValue_));
	infileBin.read((char *)&maxPosition_.x, sizeof(maxPosition_.x));
	infileBin.read((char *)&maxPosition_.y, sizeof(maxPosition_.y));
	infileBin.read((char *)&width_, sizeof(width_));
	infileBin.read((char *)&height_, sizeof(height_));

	infileBin.read((char *)&row_, sizeof(row_));
	infileBin.read((char *)&col_, sizeof(col_));
	data_.create(row_, col_, CV_32FC1);
	for (int h = 0; h < row_; h++)
		for (int w = 0; w < col_; w++)
		{
			float tmp;
			infileBin.read((char *)&tmp, sizeof(tmp));
			data_.at<float>(h, w) = tmp;
		}

	infileBin.close();
}

void CimgConvolution::showAsImg()
{
	for (int k = 0; k < data_.cols; k++)
	{
		Mat img_kernel(height_, width_, CV_32FC1);
		for (int h = 0; h < height_; h++)
		{
			for (int w = 0; w < width_; w++)
				img_kernel.at<float>(h, w) = data_.at<float>(w + h*width_, k);
		}

		char tmp[10];
		sprintf(tmp, "%d", k);
		string show = "img_kernel";
		show += tmp;
		imshow(show, img_kernel);
	}

	cvWaitKey(0);
}