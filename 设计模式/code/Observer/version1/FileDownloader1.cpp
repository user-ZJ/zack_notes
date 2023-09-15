class FileDownloader
{
	string m_filePath;
	int m_fileNumber;
	ProgressBar* m_progressBar;//UI组件 ProgressBar-> Control -> Object

public:
	FileDownloader(const string& filePath, 
	int fileNumber, 
	ProgressBar* progressBar) :
		m_filePath(filePath), 
		m_fileNumber(fileNumber),
		m_progressBar(progressBar)
	{
	}
	void download(){

		//1.网络下载准备

		//2.文件流处理

		//3.设置进度条
		for (int i = 0; i < m_fileNumber; i++){
			//...
			float progressValue = m_fileNumber;
			progressValue = (i + 1) / progressValue;
			
			m_progressBar->setValue(progressValue);//抽象责任：进度通知
		}
		//4.文件存储
	}
};