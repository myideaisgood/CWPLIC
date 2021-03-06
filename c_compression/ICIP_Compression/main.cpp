#include "arithmetic_codec.h"
#include "encode.h"
#include "decode.h"
#include "stdio.h"
#include <time.h>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <iterator>
#include <windows.h>

typedef std::vector<std::string> stringvec;

void read_directory(const std::string& name, stringvec& v)
{
	std::string pattern(name);
	pattern.append("\\*");
	WIN32_FIND_DATA data;
	HANDLE hFind;
	if ((hFind = FindFirstFile(pattern.c_str(), &data)) != INVALID_HANDLE_VALUE) {
		do {
			v.push_back(data.cFileName);
		} while (FindNextFile(hFind, &data) != 0);
		FindClose(hFind);
	}
}

void printUsage(char *s) {
	printf("Usage: %s e [source file (ppm)] [compressed file (bin)]\n", s);
	printf("Usage: %s d [compressed file (bin)] [decoded file (ppm)]\n", s);
}

int main(int argc, char *argv[]) {

	char weights_y[40], weights_u[40], weights_v[40];
	strcpy(weights_y, "weights_y.txt");
	strcpy(weights_u, "weights_u.txt");
	strcpy(weights_v, "weights_v.txt");

	if (argc == 4 && argv[1][0] == 'e') {

		std::string codename = argv[3];
		std::string code(".bin");
		std::string code_y, code_u, code_v;
		size_t pos = codename.find(code);
		code_y = codename.replace(pos, codename.length(), "_y.bin");
		code_u = codename.replace(pos, codename.length(), "_u.bin");
		code_v = codename.replace(pos, codename.length(), "_v.bin");

		char *chr_y = _strdup(code_y.c_str());
		char *chr_u = _strdup(code_u.c_str());
		char *chr_v = _strdup(code_v.c_str());

		float bpp = 0;

		bpp = runEncoder(argv[2], chr_y, chr_u, chr_v, weights_y, weights_u, weights_v);
	}
	else if (argc == 4 && argv[1][0] == 'd') {

		std::string codename = argv[2];
		std::string code(".bin");
		std::string code_y, code_u, code_v;
		size_t pos = codename.find(code);
		code_y = codename.replace(pos, codename.length(), "_y.bin");
		code_u = codename.replace(pos, codename.length(), "_u.bin");
		code_v = codename.replace(pos, codename.length(), "_v.bin");

		char *chr_y = _strdup(code_y.c_str());
		char *chr_u = _strdup(code_u.c_str());
		char *chr_v = _strdup(code_v.c_str());

		runDecoder(chr_y, chr_u, chr_v, argv[3], weights_y, weights_u, weights_v);
	}
	else {
		printUsage(argv[0]);
	}
}