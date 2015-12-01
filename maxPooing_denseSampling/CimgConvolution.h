#pragma once

#include"opencv.h"

class CimgConvolution
{
public:
	float maxValue_;
	Point maxPosition_;
	int width_;
	int height_;
	Mat data_;
	int row_;
	int col_;

public:
	void save(string fileName);
	void load(string fileName);
	void showAsImg();
};