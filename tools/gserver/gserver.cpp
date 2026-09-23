/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(_WIN64)
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#endif

#define PORT		1221
#define BUFFER_SIZE	512

	// const int depth = 7, m = 1 << 17;
	// int B_PL_prev = 0, b_prev = 0;
	// for (int b = 500000000; b <= 510000000; b += 2)
	// {
	// 	const size_t esize = size_t(m * log2(b)) + 1;
	// 	const int B_PL = static_cast<int>((esize - 1) >> depth) + 1;
	// 	if (B_PL != B_PL_prev)
	// 	{
	// 		if (B_PL_prev != 0) std::cout << b << ": " << B_PL << ", " << b - b_prev << std::endl;
	// 		B_PL_prev = B_PL; b_prev = b;
	// 	}
	// }

static void run()
{
	SOCKET server_socket = socket(AF_INET, SOCK_STREAM, 0);
	if (server_socket == INVALID_SOCKET)
	{
		throw std::runtime_error("cannot open server socket");
	}

	const int enabled = 1;
	if (setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, (const char *)&enabled, sizeof(enabled)) != 0)
	{
		throw std::runtime_error("setsockopt failed");
	}

	struct sockaddr_in server_addr;
	server_addr.sin_family = AF_INET;
	server_addr.sin_port = htons(PORT);
	server_addr.sin_addr.s_addr = INADDR_ANY;
	if (bind(server_socket, (struct sockaddr *)&server_addr, sizeof(server_addr)) == SOCKET_ERROR)
 	{
		throw std::runtime_error("bind failed");
	}

	if (listen(server_socket, SOMAXCONN) == SOCKET_ERROR)
	{
		throw std::runtime_error("listen failed");
	}

	std::cout << "Server is listening on port " << PORT << "..." << std::endl;

	size_t task_id = 0;

	while (true)
	{
		SOCKET socket = accept(server_socket, nullptr, nullptr);
		if (socket == INVALID_SOCKET)
		{
			throw std::runtime_error("accept failed");
		}

		char buffer[BUFFER_SIZE];
		const ssize_t size = recv(socket, buffer, BUFFER_SIZE, 0);
		if (size > 0)
		{
			std::cout << "Task " << task_id << ": '" << buffer << "'." << std::endl;
			std::vector<std::string> token;
			std::stringstream ssl(buffer);
			std::string item; while (std::getline(ssl, item, ' ')) token.push_back(item);
			if (token.size() == 4)
			{
				int n = std::stoi(token[0]), b = std::stoi(token[1]);
				std::string res_path = token[2], proof_path = token[3];

				std::ostringstream sse; sse << "C:\\genefer\\genefer22g.exe -p -n " << n << " -b " << b << " -f gproof";
				sse << std::endl << "results.txt => " << res_path << ", gproof.proof => " << proof_path;
				std::cout << sse.str() << std::endl;
			}

			memset(buffer, 0, BUFFER_SIZE);
			strcpy(buffer, "NOK");
			send(socket, buffer, size, 0);
			close(socket);
			std::cout << "Task " << task_id << " terminated." << std::endl;

			++task_id;
		}

		// std::thread t([=]()
		// {
		// 	bool alive = true;
		// 	while (alive)
		// 	{
		// 		char buffer[BUFFER_SIZE];
		// 		const ssize_t size = recv(socket, buffer, BUFFER_SIZE, 0);
		// 		if (size > 0)
		// 		{
		// 			std::cout << "New task: '" << buffer << "'." << std::endl;
		// 			send(socket, buffer, size, 0);	// echo

		// 			std::ostringstream sse; sse << "C:\\genefer\\genefer22g.exe -p -n " << n << " -b " << b << " -f gproof";
		// 			std::system(sse.str().c_str());
		// 		}
		// 		else alive = false;
		// 	}

		// 	std::cout << "Connection closed." << std::endl;

		// 	close(socket);
		// }); t.detach();
	}

	close(server_socket);
}

int main()
{
#if defined(_WIN64)
	WSADATA wsaData;
	WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif

	try
	{
		run();
	}
	catch (const std::runtime_error & e)
	{
		std::cerr << "Error: " << e.what() << "." << std::endl << std::flush;
#if defined(_WIN64)
		WSACleanup();
#endif
		return EXIT_FAILURE;
	}

#if defined(_WIN64)
	WSACleanup();
#endif
	return EXIT_SUCCESS;
}
