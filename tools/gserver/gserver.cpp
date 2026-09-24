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
#include <chrono>
#include <vector>
#include <list>
#include <mutex>
#include <filesystem>
#include <cmath>

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
typedef int SOCKET;
#define INVALID_SOCKET	(SOCKET)(~0)
#define SOCKET_ERROR	(-1)
#endif

#define PORT		1221
#define BUFFER_SIZE	512

const int depth = 7;

class Task
{
private:
	size_t _id;
	SOCKET _socket;
	int _n, _b;
	std::string _res_path, _proof_path;

public:
	Task() {}
	Task(const size_t id, const SOCKET & socket, const int n, const int b, const std::string & res_path, const std::string & proof_path)
		: _id(id), _socket(socket), _n(n), _b(b), _res_path(res_path), _proof_path(proof_path) {}

	size_t id() const { return _id; }
	const SOCKET & socket() const { return _socket; }
	int n() const { return _n; }
	int b() const { return _b; }
	const std::string & res_path() const { return _res_path; }
	const std::string & proof_path() const { return _proof_path; }
};

typedef std::pair<int, Task> PTask;
std::list<PTask> tasks;
std::mutex tasks_mutex;

static int B_PietrzakLi(const int n, const int b)
{
	const size_t esize = size_t((1 << n) * log2(double(b))) + 1;
	return int((esize - 1) >> depth) + 1;
}

static void compute()
{
	Task task;

	while (true)
	{
		bool found = false;
		{
			const std::lock_guard<std::mutex> lock(tasks_mutex);
			if (!tasks.empty())
			{
				std::cout << tasks.size() << " tasks:";
				for (const auto & p : tasks) std::cout << " " << p.first;
				std::cout << std::endl;

				task = tasks.front().second;
				tasks.pop_front();
				found = true;
			}
		}

		if (!found) std::this_thread::sleep_for(std::chrono::milliseconds(100));
		else
		{
			std::ostringstream sse; sse << "C:\\genefer\\genefer22g.exe -p -n " << task.n() << " -b " << task.b() << " -f gproof";
			std::system(sse.str().c_str());

			std::filesystem::rename("results.txt", task.res_path());
			std::filesystem::rename("gproof.proof", task.proof_path());

			char buffer[BUFFER_SIZE];
			memset(buffer, 0, BUFFER_SIZE);
			strcpy(buffer, "OK");
			send(task.socket(), buffer, BUFFER_SIZE, 0);
			close(task.socket());

			std::cout << "Task " << task.id() << " terminated." << std::endl;

			std::filesystem::remove("results.txt");
			std::filesystem::remove("gproof.proof");
		}
	}
}

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

	std::thread t(compute); t.detach();

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
				const int n = std::stoi(token[0]), b = std::stoi(token[1]);
				const std::lock_guard<std::mutex> lock(tasks_mutex);
				tasks.push_back(std::make_pair(B_PietrzakLi(n, b), Task(task_id, socket, n, b, token[2], token[3])));
				tasks.sort([](const PTask & x, const PTask & y) { return (x.first < y.first); });
			}

			++task_id;
		}
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
