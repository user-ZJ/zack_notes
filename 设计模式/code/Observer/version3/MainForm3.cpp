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

	
		downloader.addIProgress(make_shared<ConsoleNotifier>()); 
		downloader.addIProgress(shared_from_this()); 

		downloader.download();

		downloader.removeIProgress(ip);

	}

	void DoProgress(float value) override{
		progressBar->setValue(value);
	}
};

// A ---> shared_ptr<B>

class ConsoleNotifier : public IProgress {
public:
	void DoProgress(float value) override {
		cout << ".";
	}
};

