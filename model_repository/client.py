import torch
import tritonclient.http as httpclient
from sklearn.datasets import load_digits


# preprocessing function
def extract_data():
    img, _ = load_digits(return_X_y=True)
    img = (
        torch.tensor(img[0], device=torch.device('cpu'))
        .reshape((1, 64))
        .to(dtype=torch.float32)
    )

    return img.numpy()


if __name__ == '__main__':
    transformed_img = extract_data()
    gold_outpus = [
        [
            b'283.724365:0',
            b'147.216019:9',
            b'110.719009:5',
            b'60.232311:8',
            b'-4.155471:2',
            b'-21.064369:3',
            b'-52.824787:7',
            b'-73.959213:6',
            b'-137.592606:4',
            b'-251.596436:1',
        ]
    ]
    # Setting up client
    client = httpclient.InferenceServerClient(url="localhost:8000")

    inputs = httpclient.InferInput("input__0", transformed_img.shape, datatype="FP32")
    inputs.set_data_from_numpy(transformed_img, binary_data=False)

    outputs = httpclient.InferRequestedOutput(
        "output__0", binary_data=False, class_count=10
    )

    # Querying the server
    results = client.infer(model_name="pt-dense", inputs=[inputs], outputs=[outputs])
    inference_output = results.as_numpy("output__0")
    print('OUTPUTS', inference_output)
    print()
    print('TRUE OUTPUTS', gold_outpus)
