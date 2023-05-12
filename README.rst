.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/columbia-skip-doc.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/columbia-skip-doc
    .. image:: https://readthedocs.org/projects/columbia-skip-doc/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://columbia-skip-doc.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/columbia-skip-doc/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/columbia-skip-doc
    .. image:: https://img.shields.io/pypi/v/columbia-skip-doc.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/columbia-skip-doc/
    .. image:: https://img.shields.io/conda/vn/conda-forge/columbia-skip-doc.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/columbia-skip-doc
    .. image:: https://pepy.tech/badge/columbia-skip-doc/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/columbia-skip-doc
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/columbia-skip-doc

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

=================
columbia-skip-doc
=================

******************
Setup instructions
******************

The repo contains a file called conda_environment_py39.yaml that can be used to create a conda environment with all the necessary
dependencies to run the code. The environment can be created by running the following command:

$ conda env create -f conda_environment_py39.yaml

This Anaconda environment uses Python 3.9.16. Depending on your conda version, activate the environment by running:

$ conda activate columbia-skip-doc

Note: the yaml file containing the environment assumes a Anaconda installation at a standard location: 

$ /usr/local/anaconda3

If your Anaconda installation is not at this location, please modify the yaml file accordingly in order to generate the environment.
However, please do not commit that change to the repo.

With the activated environment, you should be able to run Streamlit by running the following command:

$ streamlit hello

This will open a tab in your browser to give you an idea of Streamlit. Once you're familiarized, you can run the app by running:

$ streamlit run streamlit_app.py

Note: this needs to happen from the $project_root/src/columbia_skip_doc/ directory.

Additionally, there is a requirements.txt file that can be used to install the necessary dependencies in a virtual environment if that
is the approach you prefer.

******************
Usage instructions
******************

Currently, the backend is decoupled from the Streamlit app. That is due to the need for compute in order to execute our prompt learning
pipeline and generate a model ready for inferencing. Therefore, the backend.py file should be run separately in an environment
that containsn a GPU. 

For example, we used credicts on GCP (Google Cloud Platform). We provisioned an instance from the VM Instances marketplace.
The VM we used was provisioned using the the template from GCP marketplace called "Deep Learning VM Image". This template
can be found and created from the following link: https://console.cloud.google.com/marketplace/vm/config/click-to-deploy-images/deeplearning

Once the VM has been provisioned, accessed and the environment defined in either our conda yaml file or the requirements.txt file,
the backend can be run by executing the following command:

$ python backend.py --use_cuda=True

Executing the above command will run the data pipeline based on the MedQuAD dataset. The pipeline will then execute our prompt
engineering pipeline built on top of the OpenPrompt framework, generate a prompt-tuned PubMED based GPT2 model, save the model artifact
locally and save the run's details.

******************
Docker Container
******************

The Skip-Doc repo contains a Dockerfile (src/columbia_skip_doc) that is used to build a docker container image containing all necessary files and dependencies
to run the web application. It also exposes the Streamlit default port of 8501 and provides a entrypoint to run the streamlit python file.

The docker image container can be built by running the following command:

$ docker build -t app_name .

Once the image is built, the container can be downloaded to disk or uploaded to desired container registry. The container can be run by executing the following
command:

$ docker run -p 8501:8501 app_name

Running the above command will launch the container and expose the port 8501 to the host machine. The model will be loaded onto CPU as default. 
If running the container from within a CUDA-enabled VM, the container can be run by executing the following command:

$ docker run --runtime=nvidia -p 8501:8501 app_name -- --use_cuda

When the application launches successully, a URL will be provided in the terminal. Copy and paste the URL into a browser to access the application.

******************
Application Usage
******************

When the web application first loads, title information and underlying processor (CPU vs GPU) will be displayed along a button to begin the Q/A session. 
Clicking the button will begin a series of 9 questions to ascertain relevent details about the patient's condition, which are then used to generate
an optimized input prompt to query the model.

After the last question is answered, the application will display a loading message while the model is queried. Once the model has finished, the 
input prompt and the model's response will be displayed. The user can then click the "Restart" button to begin another Q/A session.

******************
References
******************

@article{ding2021openprompt,
  title={OpenPrompt: An Open-source Framework for Prompt-learning},
  author={Ding, Ning and Hu, Shengding and Zhao, Weilin and Chen, Yulin and Liu, Zhiyuan and Zheng, Hai-Tao and Sun, Maosong},
  journal={arXiv preprint arXiv:2111.01998},
  year={2021}
}

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/
