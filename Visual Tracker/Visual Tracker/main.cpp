// Visual Tracker.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include<iostream>
using namespace std;

#include"Visual Tracker.h"


int main()
{
	cout << "------------------Visual Tracker--------------------" << endl;
	//cout << "Please Choose Video:" << endl;

	string path = "D:\\file_zhang\\VOT_data\\vot2016\\ball1";
	Video seq(path);
	seq.init();

	Visual_Tracker Tracking(&seq);

	cout << "Please Choose Tracker:" << endl;
	cout << "0. quit" << endl;
	cout << "1. video tracking" << endl;
	vector<bbox> result;

	int choose_number;
	cin >> choose_number;
	switch (choose_number)
	{
	case 0:
		exit(0);
	case 1:
		result=Tracking.run(choose_number,true);
		break;
	}

	return 0;
}

