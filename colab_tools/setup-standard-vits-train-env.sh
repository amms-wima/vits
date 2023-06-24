ln -s /content/drive/MyDrive/ ~/prj
cd /content/colab_tools
./setup-build-env.sh -g -csmf /content/drive/MyDrive/vits/build /content/vits
./setup-dataset.sh -d -s -f /content/colab_tools/dataset-list.csv /content/vits
./check-file-list.sh -f /content/colab_tools/critical-file-list.csv /content/vits