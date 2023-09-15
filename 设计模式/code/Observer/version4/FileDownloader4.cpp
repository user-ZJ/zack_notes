#include <iostream>
#include <list>
#include <memory>


using namespace std;

class IProgress {
public:
  virtual void DoProgress(float value) = 0;
  virtual ~IProgress() {}
};

class Subject {
private:
  list<weak_ptr<IProgress>> m_iprogressList; // 弱引用
protected:
  virtual void onProgress(float value) {
    for (auto &progress : m_iprogressList) {
      // 弱引用检查生命周期是否存在？
      shared_ptr<IProgress> s_progress = progress.lock();
      if (s_progress != nullptr) {
        s_progress->DoProgress(value);
      }
    }

    // 记住删除空的弱引用!
    for (auto iter = m_iprogressList.begin(); iter != m_iprogressList.end();) {
      if ((*iter).expired()) {
        iter = m_iprogressList.erase(iter);
      } else {
        iter++;
      }
    }
  }

public:
  void addIProgress(const shared_ptr<IProgress> &iprogress) {
    m_iprogressList.push_back(iprogress);
  }
  void removeIProgress(const shared_ptr<IProgress> &iprogress) {
    for (auto iter = m_iprogressList.begin(); iter != m_iprogressList.end();) {
      if ((*iter).expired() || (*iter).lock() == iprogress) {
        iter = m_iprogressList.erase(iter);
      } else {
        iter++;
      }
    }
  }
};

class FileDownloader : public Subject {
  string m_filePath;
  int m_fileNumber;

public:
  FileDownloader(const string &filePath, int fileNumber)
      : m_filePath(filePath), m_fileNumber(fileNumber) {}
  void download() {
    // 1.下载动作

    // 2.设置进度
    for (int i = 0; i < m_fileNumber; i++) {
      //...
      float progressValue = m_fileNumber;
      progressValue = (i + 1) / progressValue;
      onProgress(progressValue); // 通知观察者
    }
  }
};
