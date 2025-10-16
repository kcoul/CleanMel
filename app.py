import torch
import spaces
import tempfile
import soundfile as sf
import gradio as gr
import librosa as lb
import yaml
import numpy as np
import matplotlib.pyplot as plt
from model.arch.cleanmel import CleanMel
from model.vocos.offline.pretrained import Vocos
from model.io.stft import InputSTFT, TargetMel
from huggingface_hub import hf_hub_download

DEVICE = torch.device("cpu") #"cuda"

def read_audio(file_path):
    audio, sample_rate = sf.read(file_path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sample_rate != 16000:
        audio = lb.resample(audio, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    return torch.tensor(audio).float().squeeze().unsqueeze(0)

def stft(audio):
    transform = InputSTFT(
            n_fft=512,
            n_win=512,
            n_hop=128,
            normalize=False,
            center=True,
            onesided=True,
            online=False
        ).eval().to(DEVICE)
    return transform(audio)

def mel_transform(audio, X_norm):
    transform = TargetMel(
        sample_rate=16000,
        n_fft=512,
        n_win=512,
        n_hop=128,
        n_mels=80,
        f_min=0,
        f_max=8000,
        power=2,
        center=True,
        normalize=False,
        onesided=True,
        mel_norm="slaney",
        mel_scale="slaney",
        librosa_mel=True,
        online=False
    ).eval().to(DEVICE)
    return transform(audio, X_norm)

def load_cleanmel(model_name):
    model_config = f"./configs/model/cleanmel_offline.yaml"
    model_config = yaml.safe_load(open(model_config, "r"))["model"]["arch"]["init_args"]
    cleanmel = CleanMel(**model_config)
    REPO_ID = "WestlakeAudioLab/CleanMel"
    arch_ckpt = hf_hub_download(repo_id=REPO_ID, filename=f"ckpts/CleanMel/{model_name}")
    cleanmel.load_state_dict(torch.load(arch_ckpt))
    return cleanmel.eval()

def load_vocos():
    vocos = Vocos.from_hparams(config_path="./configs/model/vocos_offline.yaml")
    REPO_ID = "WestlakeAudioLab/CleanMel"
    vocos_ckpt = hf_hub_download(repo_id=REPO_ID, filename="ckpts/Vocos/vocos_offline.pt")
    vocos = Vocos.from_pretrained(None, vocos_ckpt, model=vocos)
    return vocos.eval()

def get_mrm_pred(Y_hat, x, X_norm):
    X_noisy = mel_transform(x, X_norm)
    Y_hat = Y_hat.squeeze()
    Y_hat = torch.square(Y_hat * (torch.sqrt(X_noisy) + 1e-10))
    return Y_hat

def safe_log(x):           
    return torch.log(torch.clip(x, min=1e-5)) 

def output(y_hat, logMel_hat):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        sf.write(tmp_file.name, y_hat.squeeze().cpu().numpy(), 16000)
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp_logmel_np_file:
        np.save(tmp_logmel_np_file.name, logMel_hat.squeeze().cpu().numpy())
    logMel_img = logMel_hat.squeeze().cpu().numpy()[::-1, :]
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_logmel_img:
        # give a plt figure size according to the logMel shape
        plt.figure(figsize=(logMel_img.shape[1] / 100, logMel_img.shape[0] / 50))
        plt.clf()
        plt.imshow(logMel_img, vmin=-11, cmap="jet")
        plt.tight_layout()
        plt.ylabel("Mel bands")
        plt.xlabel("Time (second)")
        plt.yticks([0, 80], [80, 0])
        dur = y_hat.shape[-1] / 16000
        xticks = [int(x) for x in np.linspace(0, logMel_img.shape[-1], 11)]
        xticks_str = ["{:.1f}".format(x) for x in np.linspace(0, dur, 11)]
        plt.xticks(xticks, xticks_str)
        plt.savefig(tmp_logmel_img.name)
    
    return tmp_file.name, tmp_logmel_img.name, tmp_logmel_np_file.name

@spaces.GPU
@torch.inference_mode()
def enhance_cleanmel_L_mask(audio_path):
    model = load_cleanmel("offline_CleanMel_L_mask.ckpt").to(DEVICE)
    vocos = load_vocos().to(DEVICE)
    x = read_audio(audio_path).to(DEVICE)
    X, X_norm = stft(x)
    Y_hat = model(X, inference=True)
    MRM_hat = torch.sigmoid(Y_hat)
    Y_hat = get_mrm_pred(MRM_hat, x, X_norm)
    logMel_hat = safe_log(Y_hat)
    y_hat = vocos(logMel_hat, X_norm).clamp(min=-1, max=1)
    return output(y_hat, logMel_hat)

@spaces.GPU
@torch.inference_mode()
def enhance_cleanmel_L_map(audio_path):
    model = load_cleanmel("offline_CleanMel_L_map.ckpt").to(DEVICE)
    vocos = load_vocos().to(DEVICE)
    x = read_audio(audio_path).to(DEVICE)
    X, X_norm = stft(x)
    logMel_hat = model(X, inference=True)
    y_hat = vocos(logMel_hat, X_norm).clamp(min=-1, max=1)
    return output(y_hat, logMel_hat)

def reset_everything():
    """Reset all components to initial state"""
    return None, None, None

if __name__ == "__main__":
    demo = gr.Blocks()
    with gr.Blocks(title="CleanMel Demo") as demo:
        gr.Markdown("## CleanMel Demo")
        gr.Markdown("This demo showcases the CleanMel model for speech enhancement.")
        
        with gr.Row():
            audio_input = gr.Audio(label="Input Audio", type="filepath", sources="upload")
            with gr.Column():
                enhance_button_map = gr.Button("Enhance Audio (offline CleanMel_L_map)")
                enhance_button_mask = gr.Button("Enhance Audio (offline CleanMel_L_mask)")
                clear_btn = gr.Button(
                    "🗑️ Clear All",
                    variant="secondary",
                    size="lg"
                )
        
        output_audio = gr.Audio(label="Enhanced Audio", type="filepath")
        output_mel = gr.Image(label="Output LogMel Spectrogram", type="filepath", visible=True)
        output_np = gr.File(label="Enhanced LogMel Spec. (.npy)", type="filepath")
        
        enhance_button_map.click(
            enhance_cleanmel_L_map, 
            inputs=audio_input, 
            outputs=[output_audio, output_mel, output_np]
        )
        
        enhance_button_mask.click(
            enhance_cleanmel_L_mask, 
            inputs=audio_input, 
            outputs=[output_audio, output_mel, output_np]
        )
        clear_btn.click(
                fn=reset_everything,
                outputs=[output_audio, output_mel, output_np]
        )

    demo.launch(debug=False, share=True)