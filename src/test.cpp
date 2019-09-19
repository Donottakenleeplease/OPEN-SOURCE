#include <stdio.h>
#include <string.h>
#include <time.h>
#include <iostream>
 
#include "mongoose.h"
#include "base64.h"
 
using namespace std;
 
int env_handler(struct mg_connection *conn);
 
int main(int argc, char *argv[])
{
  struct mg_server* server;
  server = mg_create_server(NULL);                  // 初始化一个mongoose server
  mg_set_option(server, "listening_port", "8003");  // 设置端口号为8003
  mg_add_uri_handler(server, "/", env_handler);     // 设置回调函数
  printf("Starting on port %s ...\n", mg_get_option(server, "listening_port"));
  while (1) {
    mg_poll_server(server, 100);  // 超时时间（ms）
  }
  mg_destroy_server(&server);
  
  return 0;
}
 
int env_handler(struct mg_connection *conn) 
{
  static int counter = 0;
  counter++;

  const char* encoded_data = conn->content;  // 服务端收到的消息
  int encoded_len = conn -> content_len;
  string str_encoded(encoded_data,encoded_len);
  printf("counter:%d,%s\n",counter,str_encoded.c_str());

  vector<BYTE> str_decoded_byte = base64_decode(str_encoded);
  int decoded_len = str_decoded_byte.size();
  string str_decoded;
  str_decoded.assign(str_decoded_byte.begin(),str_decoded_byte.end());
  mg_printf(conn,"Received:%s,%d",str_decoded.c_str(),counter);

  return 0;
}
