/*
 * シリアルポート受信サンプルプログラム
 * Version 1.0  2006.10.19  Y.Ebihara (SiliconLinux)
 *
 * このプログラムはシリアルポートをopenして、データを16進数表示する
 * サンプルプログラムです。
 *   i386パソコン、およびCAT760で動作検証をしています。
 *
 * test-machine: TSR-V2, i386PC, CAT760
*/


#include "IMU.h"


// シリアルポートの初期化
void IMU::serial_init()
{
    fd = open(DEV_NAME,O_RDWR);
    if(fd<0){
        // デバイスの open() に失敗したら
        //perror(argv[1]);
        // exit(1);

        std::cout<<"Cannot open serial port!"<<std::endl;
    }

    struct termios tio;
    memset(&tio,0,sizeof(tio));
    tio.c_cflag = CS8 | CLOCAL | CREAD;
    tio.c_cc[VTIME] = 100;
    // ボーレートの設定
    cfsetispeed(&tio,BAUD_RATE);
    cfsetospeed(&tio,BAUD_RATE);
    // デバイスに設定を行う
    tcsetattr(fd,TCSANOW,&tio);
}


std::vector<std::string> split(std::string str, char del)
{
    int first = 0;
    int last = str.find_first_of(del);

    std::vector<std::string> result;

    while (first < str.size()) {
        std::string subStr(str, first, last - first);

        result.push_back(subStr);

        first = last + 1;
        last = str.find_first_of(del, first);

        if (last == std::string::npos) {
            last = str.size();
        }
    }

    return result;
}

std::vector<float> IMU::getData()
{
        int i;
        int len;
        char buffer[BUFF_SIZE];

        len=read(fd,buffer,BUFF_SIZE);

        if(len==0){
            // read()が0を返したら、end of file
            // 通常は正常終了するのだが今回は無限ループ
            //continue;
            std::cout<<"No data on serial port!"<<std::endl;
        }
        if(len<0){
            std::cout<<"IMU error - data length < 0"<<std::endl;
            //printf("%s: ERROR\n",argv[0]);
            // read()が負を返したら何らかのI/Oエラー
            //perror("");
            //exit(2);
        }

        // char buffer to string
        std::string sbuffer = buffer;
        char separator = ',';

        std::vector<std::string> separatedVector = split(sbuffer,separator);
        std::vector<float> imuData;
        //float imuData[DATA_SIZE];

        int n = 0;
        for( std::vector<std::string>::iterator i = separatedVector.begin(); i != separatedVector.end(); i++/*,n++*/ )
        {
            //std::cout<< *i;
            //imuData[n] = std::stof(*i);
            //std::cout << imuData[n] << "    ";
           // std::cout<<*i<<"   ";
            imuData.push_back(std::stof(*i));
            std::cout<<imuData.back()<<"  ";
        }

        std::cout<<std::endl;

        return imuData;
}
