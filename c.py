import tensorflow as tf
import os
import numpy as np

# Convert pcap file to bytes
def pcap_to_bytes(pcap_file):
    with open(pcap_file, 'rb') as f:
        return f.read()

# Get list of pcap files in directory
pcap_dir_password_attack = './password_attack/'
pcap_files_password_attack = [os.path.join(pcap_dir_password_attack, f) for f in os.listdir(pcap_dir_password_attack) if f.endswith('.pcap')]
pcap_files_password_attack = pcap_files_password_attack[:300]
label_pa = np.full(len(pcap_files_password_attack), 0)

pcap_dir_port_scan = './port_scan/'
pcap_files_port_scan = [os.path.join(pcap_dir_port_scan, f) for f in os.listdir(pcap_dir_port_scan) if f.endswith('.pcap')]
pcap_files_port_scan = pcap_files_port_scan[:300]
pcap_files = pcap_files_password_attack + pcap_files_port_scan
label_ps = np.full(len(pcap_files_port_scan), 1)
# Convert pcap files to bytes
pcap_bytes = [pcap_to_bytes(f) for f in pcap_files]
# pcap_bytes = [np.fromfile(f, dtype=np.uint8) for f in pcap_files]

# Define labels
labels = np.concatenate((label_pa, label_ps), axis=0)
# labels = np.concatenate((label_pa, label_ps), axis=0).astype(np.int64)

print(labels.shape)
print(len(pcap_bytes))
print(len(pcap_bytes[0]))

# Convert pcap bytes to numpy array
max_length = max(len(b) for b in pcap_bytes)
pcap_bytes_array = np.zeros((len(pcap_bytes), max_length))
print(max_length)
for i, b in enumerate(pcap_bytes):
    padded_b = b.ljust(max_length, b'\x00')
    pcap_bytes_array[i, :] = np.frombuffer(padded_b, dtype=np.uint8)


# Define TensorFlow model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(max_length,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(pcap_bytes_array, labels, test_size=0.2, random_state=42)
# Train the model
model.fit(X_train, y_train,validation_data=(X_test,y_test), epochs=10, batch_size=32)

model.save('model1.h5')