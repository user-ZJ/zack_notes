class MainForm : public Form, public IProgress,
	public enable_shared_from_this<MainForm>
{
	shared_ptr<TextBox> txtFilePath;
	shared_ptr<TextBox> txtFileNumber;
	shared_ptr<ProgressBar> progressBar;

public:
	void Button1_Click(){

		string filePath = txtFilePath->getText();
		int number = atoi(txtFileNumber->getText().c_str());


		FileDownloader downloader(filePath, number);

		shared_ptr<IProgress> ip1=make_shared<ConsoleNotifier>();
		downloader.setProgress( ip1);


		//shared_ptr<IProgress> ip2{this}; 错误！
		shared_ptr<IProgress> ip2=shared_from_this(); 
		downloader.setProgress( ip2); 

		downloader.download();
	}

	void DoProgress(float value) override {
		progressBar->setValue(value);
	}
};

class ConsoleNotifier : public IProgress {
public:
	void DoProgress(float value) override{
		cout << ".";
	}
};

