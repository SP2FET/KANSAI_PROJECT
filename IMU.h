
#include <iostream>
#include <string>
#include <vector>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <termios.h>
#include <time.h>


#define DEV_NAME    "/dev/ttyACM0"        // デバイスファイル名
#define BAUD_RATE    B9600                // RS232C通信ボーレート
#define BUFF_SIZE    4096                 // 適当
#define DATA_SIZE    3

class IMU
{
public:

    std::vector<float> getData();
    void serial_init();

private:
    int fd;

};
