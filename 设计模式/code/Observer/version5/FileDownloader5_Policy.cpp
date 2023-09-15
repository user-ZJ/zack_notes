#include <list>
#include <iostream>
using namespace std;



//标准库定义概念
// template< class F, class... Args >
// concept invocable =
//   requires(F&& f, Args&&... args) {
//     std::invoke(std::forward<F>(f), 
// 			    std::forward<Args>(args)...);
//   };



template<typename Observer>
class Subject
{
private:
	list<Observer>  m_progressList;
	
protected:
	void onUpdate(float value) {
		for (auto& progress : m_iprogressList){
			progress(value);
		}
	}

public:
	void addIProgress(const Observer& progress){
		m_progressList.push_back(progress);
	}
	void removeIProgress(const Observer& progress){
		m_progressList.remove(progress);
	}
};



template<typename Observer>
class FileDownloader: public Subject<Observer>
{
	string m_filePath;
	int m_fileNumber;

public:
	FileDownloader(const string& filePath, int fileNumber) :
		m_filePath(filePath), 
		m_fileNumber(fileNumber){

	}
	void download(){
		//1.下载动作

		//2.设置进度
		for (int i = 0; i < m_fileNumber; i++){
			//...
			float progressValue = m_fileNumber;
			progressValue = (i + 1) / progressValue;
			onProgress(progressValue);//通知观察者
		}

	}
};


//函数对象
struct ProgressObserver{
	void operator()(float value){
		...
	}
};

//std::functionn
using ProgressHandler = std::function<void(float)>;

//lambda
auto callback=[](float value){ 
		  ...
};


int main(){


	FileDownloader<ProgressObserver> fd2(...);//支持单一存储
	ProgressObserver pob;
	fd.addIProgress(pob);
	fd.download();


	FileDownloader<ProgressHandler> fd3;//支持多态存储（有性能成本）
	fd2.addIProgress(pob);
	fd2.addIProgress(callback);
	fd2.removeIProgress(pob);
	fd2.download();

}