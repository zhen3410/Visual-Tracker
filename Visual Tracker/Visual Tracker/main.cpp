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

	string path = "D:\\file_zhang\\VOT_data\\vot2016\\basketball";
	Video seq(path);
	seq.init();

	Visual_Tracker Tracking(&seq);

	cout << "Please Choose Tracker:" << endl;
	cout << "0. quit" << endl;
	cout << "1. video tracking" << endl;
	vector<bbox> result;
	while (1) {
		int choose_number;
		cin >> choose_number;
		switch (choose_number)
		{
		case 0:
			exit(0);
		default:
			result=Tracking.run(choose_number);
			break;
		}
	}

	system(0);
	return 0;
}

