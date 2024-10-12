#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <sstream>

using namespace std;

map<uint32_t, uint32_t> vnode2node; // 存储解析结果

void ParseNodeMapping(string filename) {
    ifstream infile(filename);
    string line;

    // 检查文件是否成功打开
    if (!infile.is_open()) {
        cerr << "无法打开文件: " << filename << endl;
        return;
    }

    // 逐行读取文件
    while (getline(infile, line)) {
        uint32_t key, value=-1;
        char arrow; // 用于读取 '->' 中的字符

        // 使用字符串流解析每一行
        stringstream ss(line);
        ss >> key >> arrow >> arrow >> value; // 读取key和value

        // 存储到map中
        vnode2node[key] = value;
    }

    infile.close(); // 关闭文件
}

int main() {
    string filename = "mix/node_mapping.txt"; // 替换为你的文件名
    ParseNodeMapping(filename);

    // 打印结果以验证
    for (const auto& pair : vnode2node) {
        cout << "vnode: " << pair.first << ", node: " << pair.second << endl;
    }

    return 0;
}