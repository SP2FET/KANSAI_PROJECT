#include <iostream>
#include "IMU.h"

IMU imu;

int main()
{
//	std::cout <<"Hell'o World!";
	imu.serial_init();
	std::vector<float> data;

	while(true)
	{
		data = imu.getData();
		///data is already displayed in getData
		//std::cout << data[0] << std::endl;


	}




	return 0;
}
