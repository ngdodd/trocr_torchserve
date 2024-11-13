# trocr_torchserve

After cloning the repository:

* Navigate to the project directory and create a virtual environment for the project: `python -m venv trocr_torchserve`
* Activate the virtual environment:
  * Windows: `.\trocr_torchserve\Scripts\activate`
  * macOS/Linux: `source trocr_torchserve/bin/activate`
* Install all required packages: `pip install -r requirements.txt`
* Run the `setup.sh` script.

Now, run the following scripts to archive the model and launch torchserve respectively:
* `archive_model.sh <model_name> <version_number>` - script that will create the torchserve .mar file for model serving. Example usage: `archive_model.sh trocr_base 1.0`
* `start_server.sh` - starts the torchserve server using configurations specified in config.properties

Once the server is running, you can run the `client.py` script which will send images from the `Teklia/IAM-line` Hugging Face dataset to the model server for inference. The usage is: `python cilent.py --ids <csv_list_of_ids>`. For example `python client.py --ids 1,5,9,13` sends images with the sample ids 1, 5, 9, and 13 from the dataset to the model server for inference.
* Note, if running the `client.py` script, each image will be displayed in a new window. Once each window in a batch is closed, the data will be sent to the server.
