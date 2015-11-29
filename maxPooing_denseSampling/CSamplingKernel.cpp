#pragma once
# include "CSamplingKernel.h"
#include <fstream>
using namespace std;

CSamplingKernel::CSamplingKernel(int width, int height)
{
	width_ = width;
	height_ = height;
	channel_ = 3;
	data_.create(height_, width_, CV_32FC3);

}

bool CSamplingKernel::getParam(int &width, int &height)
{
	width = width_;
	height = height_;

	return true;
}
bool CSamplingKernel::getData(Mat &data)
{
	data = data_.clone();

	return true;
}
void CSamplingKernel::setIndexAndLabel(int label, int index)
{
	label_ = label;
	index_ = index;
}
void CSamplingKernel::setSrcName(string name_src)
{
	name_src_ = name_src;
}

void CSamplingKernel::setData(IplImage *img, Point center)
{
	unsigned char *data_img = (unsigned char *)img->imageData;
	int width_img = img->width;
	int height_img = img->height;
	int nchannel_img = img->nChannels;
	int step_img = img->widthStep;

	int sta_x = center.x - width_ / 2;
	int end_x = center.x + width_ / 2;
	if (sta_x < 0)
	{
		sta_x = 0;
		end_x = width_;
	}
	if (end_x >= width_img)
	{
		end_x = width_img-1;
		sta_x = width_img -1 - width_;
	}

	int sta_y = center.y - height_ / 2;
	int end_y = center.y + height_ / 2;
	if (sta_y < 0)
	{
		sta_y = 0;
		end_y = height_;
	}
	if (end_y >= height_img)
	{
		end_y = height_img - 1;
		sta_y = height_img - 1 - height_;
	}

	for (int h = sta_y; h < end_y; h++)
	{
		for (int w = sta_x; w < end_x; w++)
			for (int c = 0; c < nchannel_img; c++)
		{
			data_.at<Vec3f>(h-sta_y, w-sta_x)[c] = data_img[c + w*nchannel_img + h*step_img]/255.0;
		}
	}

	//cvShowImage("ORI", img);
	//imshow("ROI", data_);
	//cvWaitKey(0);
}

float CSamplingKernel::convolution(IplImage *img, Mat &img_convolution, bool sign_expand_kernel, int conv_step)
{
	float max = -10000;

	int width_img = img->width;
	int height_img = img->height;
	int step_img = img->widthStep;
	int channel_img = img->nChannels;
	unsigned char *data_img = (unsigned char *)img->imageData;

	Mat img_mat = Mat::zeros(((width_img - width_) / conv_step + 1) * ((height_img - height_) / conv_step + 1), width_*height_, CV_32FC3);

	int h_mat = 0;
	int w_mat = 0;
	for (int h = 0; h < height_img-height_; h += conv_step)
	{
		for (int w = 0; w < width_img-width_; w += conv_step)
		{
			w_mat = 0;
			for (int hh = 0; hh < height_; hh++)
			{
				for (int ww = 0; ww < width_; ww++)
				{
					for (int cc = 0; cc < channel_img; cc++)
					{
						img_mat.at<Vec3f>(h_mat, w_mat)[cc] = data_img[cc + (w + ww)*channel_img + (h + hh)*step_img];
					}
					w_mat++;
				}
			}
			h_mat++;
		}
	}

	Mat kernel_mat;
	if (sign_expand_kernel == false)
	{
		kernel_mat.create(height_*width_, 1, CV_32FC3);
		resize(data_, kernel_mat, Size(height_*width_, 1), 0, 0, CV_INTER_LINEAR);
	}
	else
	{
		kernel_mat.create(height_*width_, 28, CV_32FC3);
		for (int h = 0; h < height_; h++)
		{
			for (int w = 0; w < width_; w++)
			{
				for (int c = 0; c < channel_; c++)
				{
					kernel_mat.at<Vec3f>(w + h*width_, 0)[c] = data_.at<Vec3f>(h, w)[c];
					kernel_mat.at<Vec3f>(w + h*width_, 1)[c] = data_.at<Vec3f>(height_-1-h, w)[c];
					kernel_mat.at<Vec3f>(w + h*width_, 2)[c] = data_.at<Vec3f>(h, width_-1-w)[c];
					kernel_mat.at<Vec3f>(w + h*width_, 3)[c] = data_.at<Vec3f>(height_-1-h, width_-1-w)[c];

					kernel_mat.at<Vec3f>(width_-1- w + h*width_, 4)[c] = data_.at<Vec3f>(h, w)[c];
					kernel_mat.at<Vec3f>(width_ - 1 - w + h*width_, 5)[c] = data_.at<Vec3f>(height_ - 1 - h, w)[c];
					kernel_mat.at<Vec3f>(width_ - 1 - w + h*width_, 6)[c] = data_.at<Vec3f>(h, width_ - 1 - w)[c];
					kernel_mat.at<Vec3f>(width_ - 1 - w + h*width_, 7)[c] = data_.at<Vec3f>(height_ - 1 - h, width_ - 1 - w)[c];

					kernel_mat.at<Vec3f>(w + (height_ - 1 - h)*width_, 8)[c] = data_.at<Vec3f>(h, w)[c];
					kernel_mat.at<Vec3f>(w + (height_ - 1 - h)*width_, 9)[c] = data_.at<Vec3f>(height_ - 1 - h, w)[c];
					kernel_mat.at<Vec3f>(w + (height_ - 1 - h)*width_, 10)[c] = data_.at<Vec3f>(h, width_ - 1 - w)[c];
					kernel_mat.at<Vec3f>(w + (height_ - 1 - h)*width_, 11)[c] = data_.at<Vec3f>(height_ - 1 - h, width_ - 1 - w)[c];


					kernel_mat.at<Vec3f>(h + w*height_, 12)[c] = data_.at<Vec3f>(h, w)[c];
					kernel_mat.at<Vec3f>(h + w*height_, 13)[c] = data_.at<Vec3f>(height_ - 1 - h, w)[c];
					kernel_mat.at<Vec3f>(h + w*height_, 14)[c] = data_.at<Vec3f>(h, width_ - 1 - w)[c];
					kernel_mat.at<Vec3f>(h + w*height_, 15)[c] = data_.at<Vec3f>(height_ - 1 - h, width_ - 1 - w)[c];

					kernel_mat.at<Vec3f>(height_ - 1 - h + w*height_, 16)[c] = data_.at<Vec3f>(h, w)[c];
					kernel_mat.at<Vec3f>(height_ - 1 - h + w*height_, 17)[c] = data_.at<Vec3f>(height_ - 1 - h, w)[c];
					kernel_mat.at<Vec3f>(height_ - 1 - h + w*height_, 18)[c] = data_.at<Vec3f>(h, width_ - 1 - w)[c];
					kernel_mat.at<Vec3f>(height_ - 1 - h + w*height_, 19)[c] = data_.at<Vec3f>(height_ - 1 - h, width_ - 1 - w)[c];

					kernel_mat.at<Vec3f>(h + (width_ - 1 - w)*height_, 20)[c] = data_.at<Vec3f>(h, w)[c];
					kernel_mat.at<Vec3f>(h + (width_ - 1 - w)*height_, 21)[c] = data_.at<Vec3f>(height_ - 1 - h, w)[c];
					kernel_mat.at<Vec3f>(h + (width_ - 1 - w)*height_, 22)[c] = data_.at<Vec3f>(h, width_ - 1 - w)[c];
					kernel_mat.at<Vec3f>(h + (width_ - 1 - w)*height_, 23)[c] = data_.at<Vec3f>(height_ - 1 - h, width_ - 1 - w)[c];

					kernel_mat.at<Vec3f>(height_ - 1 - h + (width_ - 1 - w)*height_, 24)[c] = data_.at<Vec3f>(h, w)[c];
					kernel_mat.at<Vec3f>(height_ - 1 - h + (width_ - 1 - w)*height_, 25)[c] = data_.at<Vec3f>(height_ - 1 - h, w)[c];
					kernel_mat.at<Vec3f>(height_ - 1 - h + (width_ - 1 - w)*height_, 26)[c] = data_.at<Vec3f>(h, width_ - 1 - w)[c];
					kernel_mat.at<Vec3f>(height_ - 1 - h + (width_ - 1 - w)*height_, 27)[c] = data_.at<Vec3f>(height_ - 1 - h, width_ - 1 - w)[c];
				}
			}
		}
	}

	img_convolution = img_mat * kernel_mat;
	return max;
}

void CSamplingKernel::save(string fileName)
{
	ofstream outfileBin;
	outfileBin.open(fileName, ios::binary);

	outfileBin.write((char *)&label_, sizeof(label_));
	outfileBin.write((char *)&index_, sizeof(index_));
	outfileBin.write((char *)&name_src_, sizeof(name_src_));
	outfileBin.write((char *)&width_, sizeof(width_));
	outfileBin.write((char *)&height_, sizeof(height_));
	outfileBin.write((char *)&channel_, sizeof(channel_));

	for (int h = 0; h < height_; h++)
		for (int w = 0; w < width_; w++)
			for (int c = 0; c < channel_; c++)
			{
				float tmp = data_.at<Vec3f>(h, w)[c];
				outfileBin.write((char *)&tmp, sizeof(tmp));
			}

	outfileBin.close();
}

void CSamplingKernel::saveAsImg(string fileName)
{
	IplImage *img = cvCreateImage(cvSize(width_, height_), IPL_DEPTH_8U, channel_);
	unsigned char* data = (unsigned char*)img->imageData;
	int step_img = img->widthStep;

	for (int h = 0; h < height_; h++)
		for (int w = 0; w < width_; w++)
			for (int c = 0; c < channel_; c++)
			{
				data[c + w*channel_ + h*step_img] = data_.at<Vec3f>(h, w)[c]*255.0;
			}

	cvSaveImage(fileName.c_str(), img);
	cvReleaseImage(&img);
}

void CSamplingKernel::load(string fileName)
{
	ifstream infileBin;
	infileBin.open(fileName, ios::binary);

	infileBin.read((char *)&label_, sizeof(label_));
	infileBin.read((char *)&index_, sizeof(index_));
	infileBin.read((char *)&name_src_, sizeof(name_src_));
	infileBin.read((char *)&width_, sizeof(width_));
	infileBin.read((char *)&height_, sizeof(height_));
	infileBin.read((char *)&channel_, sizeof(channel_));

	data_.create(height_, width_, CV_32FC3);
	for (int h = 0; h < height_; h++)
		for (int w = 0; w < width_; w++)
			for (int c = 0; c < channel_; c++)
			{
				float tmp = 0;
				infileBin.read((char *)&tmp, sizeof(tmp));
				data_.at<Vec3f>(h, w)[c] = tmp;
			}

	infileBin.close();
}