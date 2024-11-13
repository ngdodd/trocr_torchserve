# Use an official PyTorch image as the base image
FROM pytorch/torchserve:latest

# Set the working directory
WORKDIR /home/model-server/

# Create the model store directory
RUN mkdir -p /home/model-server/model_store

# Copy necessary files
COPY model_store/ /home/model-server/model_store/
COPY trocr_model/ /home/model-server/trocr_model/
COPY config.properties . 
COPY handler.py /home/model-server/handler.py
COPY requirements.txt .

# Install required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the ports for TorchServe
EXPOSE 8080 8081 8082

# Configure environment variables
ENV MODEL_STORE=/home/model-server/model_store
ENV TS_CONFIG_FILE=/home/model-server/config.properties

# Run TorchServe
CMD ["torchserve", "--start", "--ncs", "--model-store", "$MODEL_STORE", "--ts-config", "$TS_CONFIG_FILE"]