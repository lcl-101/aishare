#!/usr/bin/env python3
"""
JAM WebUI - A Gradio-based web interface for JAM music generation
Simple and clean implementation that follows the original JAM workflow
"""

import gradio as gr
import os
import json
import tempfile
import uuid
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# JAM imports
from omegaconf import OmegaConf
from muq import MuQMuLan
from jam.model.vae import StableAudioOpenVAE, DiffRhythmVAE
from jam.dataset import enhance_webdataset_config, DiffusionWebDataset
from safetensors.torch import load_file
from jam.model.cfm import CFM
from jam.model.dit import DiT

# ==================== JAM WebUI Utility Functions ====================

def load_model(model_config, checkpoint_path, device):
    """Load JAM CFM model from checkpoint"""
    print(f"📁 Loading model from: {checkpoint_path}")
    
    # Build CFM model from config
    dit_config = model_config["dit"].copy()
    # Add text_num_embeds if not specified - should be at least 64 for phoneme tokens
    if "text_num_embeds" not in dit_config:
        dit_config["text_num_embeds"] = 256  # Default value from DiT
    
    # Initialize DiT transformer
    dit = DiT(**dit_config)
    
    # Initialize CFM with DiT transformer
    cfm = CFM(
        transformer=dit,
        **model_config["cfm"]
    )
    
    # Move to device
    model = cfm.to(device)
    
    # Load checkpoint
    if checkpoint_path.endswith('.safetensors'):
        state_dict = load_file(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Clean state dict if needed
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        # Remove module. prefix if present
        clean_key = key.replace('module.', '') if key.startswith('module.') else key
        cleaned_state_dict[clean_key] = value
    
    # Load state dict
    model.load_state_dict(cleaned_state_dict, strict=False)
    model = model.eval()
    
    print(f"✅ Model loaded successfully with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model

def generate_latent(model, batch, sample_kwargs, negative_style_prompt_path=None, ignore_style=False):
    """Generate latent from batch data (follows infer.py pattern)"""
    with torch.inference_mode():
        batch_size = len(batch["lrc"])
        text = batch["lrc"]
        style_prompt = batch["prompt"]
        start_time = batch["start_time"]
        duration_abs = batch["duration_abs"]
        duration_rel = batch["duration_rel"]
        
        # Create zero conditioning latent
        max_frames = model.max_frames
        cond = torch.zeros(batch_size, max_frames, 64).to(text.device)
        pred_frames = [(0, max_frames)]

        default_sample_kwargs = {
            "cfg_strength": 4,
            "steps": 50,
            "batch_infer_num": 1
        }
        sample_kwargs = {**default_sample_kwargs, **sample_kwargs}
        
        # Load negative style prompt if provided
        negative_style_prompt = None
        if negative_style_prompt_path and not ignore_style:
            try:
                # Try different loading methods to handle various file formats
                import pickle
                if negative_style_prompt_path.endswith('.npy'):
                    # Handle numpy files
                    import numpy as np
                    negative_style_prompt = torch.from_numpy(np.load(negative_style_prompt_path)).to(text.device)
                else:
                    # Try standard torch.load with different protocols
                    try:
                        negative_style_prompt = torch.load(negative_style_prompt_path, map_location=text.device, weights_only=False)
                    except (pickle.UnpicklingError, AttributeError):
                        # Fallback: try with pickle protocol compatibility
                        with open(negative_style_prompt_path, 'rb') as f:
                            negative_style_prompt = torch.load(f, map_location=text.device, weights_only=False, pickle_module=pickle)
                
                if negative_style_prompt is not None:
                    if negative_style_prompt.dim() == 1:
                        negative_style_prompt = negative_style_prompt.unsqueeze(0)
                    negative_style_prompt = negative_style_prompt.repeat(batch_size, 1)
            except Exception as e:
                print(f"Warning: Could not load negative style prompt: {e}")
                negative_style_prompt = None
        
        # If negative_style_prompt is None, create a zero tensor with the same shape as style_prompt
        if negative_style_prompt is None:
            negative_style_prompt = torch.zeros_like(style_prompt)
        
        # Generate latent using CFM model
        latents, _ = model.sample(
            cond=cond,
            text=text,
            style_prompt=style_prompt,
            negative_style_prompt=negative_style_prompt,
            steps=sample_kwargs["steps"],
            cfg_strength=sample_kwargs["cfg_strength"],
            start_time=start_time,
            duration_abs=duration_abs,
            duration_rel=duration_rel,
            latent_pred_segments=pred_frames,
            batch_infer_num=sample_kwargs["batch_infer_num"]
        )

        return latents

def normalize_audio(audio, target_peak=0.9):
    """Normalize audio to target peak amplitude"""
    if isinstance(audio, torch.Tensor):
        # Get the maximum absolute value
        max_val = torch.abs(audio).max()
        
        if max_val > 0:
            # Normalize to target peak
            audio = audio * (target_peak / max_val)
        
        # Clamp to valid range
        audio = torch.clamp(audio, -1.0, 1.0)
        
    return audio

def resample_audio(audio, orig_sr, target_sr):
    """Resample audio to target sample rate"""
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        audio = resampler(audio)
    return audio

def audio_to_mono(audio):
    """Convert stereo audio to mono"""
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    return audio

def pad_or_trim_audio(audio, target_length):
    """Pad or trim audio to target length"""
    current_length = audio.shape[-1]
    
    if current_length > target_length:
        # Trim audio
        audio = audio[..., :target_length]
    elif current_length < target_length:
        # Pad audio with zeros
        pad_length = target_length - current_length
        if audio.dim() == 1:
            audio = F.pad(audio, (0, pad_length))
        else:
            audio = F.pad(audio, (0, pad_length))
    
    return audio

def ensure_audio_format(audio, target_channels=1, target_sr=44100):
    """Ensure audio has the correct format for output"""
    
    # Convert to mono if needed
    if audio.shape[0] > target_channels:
        audio = audio_to_mono(audio)
    
    # Add channel dimension if needed
    if audio.dim() == 1 and target_channels > 0:
        audio = audio.unsqueeze(0)
    
    # Normalize
    audio = normalize_audio(audio)
    
    return audio

def safe_audio_save(audio, path, sample_rate=44100):
    """Safely save audio tensor to file"""
    try:
        # Ensure correct format
        audio = ensure_audio_format(audio)
        
        # Save audio
        torchaudio.save(path, audio.cpu(), sample_rate)
        return True
        
    except Exception as e:
        print(f"Error saving audio: {e}")
        return False

def load_audio_safely(path, target_sr=None, mono=True):
    """Safely load audio file"""
    try:
        audio, sr = torchaudio.load(path)
        
        # Convert to mono if requested
        if mono:
            audio = audio_to_mono(audio)
        
        # Resample if needed
        if target_sr is not None and sr != target_sr:
            audio = resample_audio(audio, sr, target_sr)
            sr = target_sr
        
        return audio, sr
        
    except Exception as e:
        print(f"Error loading audio from {path}: {e}")
        return None, None

def get_audio_duration(audio, sample_rate):
    """Get duration of audio in seconds"""
    if audio is None:
        return 0.0
    return audio.shape[-1] / sample_rate

def create_silence(duration, sample_rate=44100, channels=1):
    """Create silent audio of specified duration"""
    num_samples = int(duration * sample_rate)
    if channels == 1:
        return torch.zeros(1, num_samples)
    else:
        return torch.zeros(channels, num_samples)

def format_duration(seconds):
    """Format duration in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"

def get_model_info(model):
    """Get information about model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
    }

def print_generation_stats(start_time, end_time, duration, model_info=None):
    """Print generation statistics"""
    generation_time = end_time - start_time
    rtf = generation_time / duration  # Real-time factor
    
    print(f"""
📊 Generation Statistics:
• Audio Duration: {format_duration(duration)}
• Generation Time: {format_duration(generation_time)}
• Real-time Factor: {rtf:.2f}x
• Speed: {'Faster' if rtf < 1 else 'Slower'} than real-time
    """)
    
    if model_info:
        print(f"""
🔧 Model Information:
• Total Parameters: {model_info['total_params']:,}
• Model Size: {model_info['model_size_mb']:.1f} MB
        """)

# ==================== JAM WebUI Main Class ====================

class JAMWebUI:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models_loaded = False
        self.cfm_model = None
        self.vae = None
        self.muq_model = None
        self.config = None
        self.base_dataset = None
        
        print(f"🎵 JAM WebUI initialized on device: {self.device}")
        print("🔄 Auto-loading models...")
        self._load_models_on_startup()
        
    def _load_models_on_startup(self):
        """Load all required models on startup"""
        if self.models_loaded:
            return
            
        try:
            print("📁 Loading configuration...")
            
            # Load configuration
            config_path = "checkpoints/jam-0.5/jam_infer.yaml"
            self.config = OmegaConf.load(config_path)
            
            # Set checkpoint path
            self.config.evaluation.checkpoint_path = "checkpoints/jam-0.5/jam-0_5.safetensors"
            
            # Resolve config variables
            OmegaConf.resolve(self.config)
            
            print("🔧 Loading VAE model...")
            
            # Load VAE (use default configuration)
            vae_type = self.config.evaluation.get('vae_type', 'diffrhythm')
            if vae_type == 'diffrhythm':
                self.vae = DiffRhythmVAE(device=self.device).to(self.device)
            else:
                self.vae = StableAudioOpenVAE().to(self.device)
            
            print("🧠 Loading CFM model...")
            
            # Load CFM model
            self.cfm_model = load_model(
                self.config.model, 
                self.config.evaluation.checkpoint_path, 
                self.device
            )
            
            print("🎨 Loading MuQ model...")
            
            # Load MuQ model for style embeddings
            try:
                # Check and fix the model file naming issue
                muq_checkpoint_dir = "checkpoints/MuQ-MuLan-large"
                safetensors_path = os.path.join(muq_checkpoint_dir, "model.safetensors")
                pytorch_path = os.path.join(muq_checkpoint_dir, "pytorch_model.bin")
                
                # If model.safetensors doesn't exist but pytorch_model.bin does, create a symlink
                if not os.path.exists(safetensors_path) and os.path.exists(pytorch_path):
                    try:
                        os.symlink("pytorch_model.bin", safetensors_path)
                        print("🔗 Created symbolic link: model.safetensors -> pytorch_model.bin")
                    except Exception as link_error:
                        print(f"⚠️ Could not create symlink: {link_error}")
                
                # Try loading from local checkpoint first
                self.muq_model = MuQMuLan.from_pretrained(
                    muq_checkpoint_dir
                ).to(self.device).eval()
                print("✅ Loaded MuQ model from local checkpoint")
            except Exception as e:
                print(f"⚠️ Failed to load local MuQ model: {e}")
                print("📥 Falling back to downloading from HuggingFace...")
                # Fallback to downloading from HuggingFace
                self.muq_model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large").to(self.device).eval()
                print("✅ Downloaded and loaded MuQ model from HuggingFace")
            
            print("📊 Setting up dataset...")
            
            # Setup base dataset
            dataset_cfg = OmegaConf.merge(self.config.data.train_dataset, self.config.evaluation.dataset)
            enhance_webdataset_config(dataset_cfg)
            dataset_cfg.multiple_styles = False
            self.base_dataset = DiffusionWebDataset(**dataset_cfg)
            
            self.models_loaded = True
            
            # Get model info
            total_params = sum(p.numel() for p in self.cfm_model.parameters())
            
            print(f"""✅ Models loaded successfully!

📊 Model Information:
- CFM Parameters: {total_params:,}
- VAE Type: {vae_type}
- Device: {self.device}
- MuQ Model: OpenMuQ/MuQ-MuLan-large

🎵 Ready to generate music!""")
            
        except Exception as e:
            print(f"❌ Failed to load models: {str(e)}")
            self.models_loaded = False
    
    def preview_lyrics(self, lyrics_file):
        """Preview uploaded lyrics file content"""
        if lyrics_file is None:
            return "Upload a lyrics file to see its content here..."
        
        try:
            # Handle both file objects and file paths
            file_path = lyrics_file.name if hasattr(lyrics_file, 'name') else lyrics_file
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Try to parse and format the lyrics nicely
            try:
                lyrics_data = json.loads(content)
                
                # Format lyrics for display
                if isinstance(lyrics_data, list) and len(lyrics_data) > 0:
                    # Standard format: [{"word": "hello", "start": 0.0, "end": 0.5}, ...]
                    if all(key in lyrics_data[0] for key in ['word', 'start', 'end']):
                        formatted_lyrics = "📝 Lyrics with Timestamps:\n\n"
                        for i, item in enumerate(lyrics_data[:50]):  # Show first 50 words
                            formatted_lyrics += f"{item['start']:.1f}s - {item['end']:.1f}s: {item['word']}\n"
                        if len(lyrics_data) > 50:
                            formatted_lyrics += f"\n... and {len(lyrics_data) - 50} more words"
                        return formatted_lyrics
                
                elif isinstance(lyrics_data, dict):
                    # Complex format like the input files: {"char": [...], "word": [...]}
                    if 'word' in lyrics_data:
                        word_data = lyrics_data['word']
                        if isinstance(word_data, list) and len(word_data) > 0:
                            formatted_lyrics = "📝 Lyrics with Timestamps:\n\n"
                            for i, item in enumerate(word_data[:50]):  # Show first 50 words
                                if isinstance(item, dict) and all(key in item for key in ['word', 'start', 'end']):
                                    formatted_lyrics += f"{item['start']:.1f}s - {item['end']:.1f}s: {item['word']}\n"
                            if len(word_data) > 50:
                                formatted_lyrics += f"\n... and {len(word_data) - 50} more words"
                            return formatted_lyrics
                    
                    # Show raw JSON structure info
                    keys = list(lyrics_data.keys())
                    return f"📋 JSON Structure:\nKeys: {', '.join(keys)}\n\nRaw content preview:\n{json.dumps(lyrics_data, indent=2, ensure_ascii=False)[:1000]}{'...' if len(str(lyrics_data)) > 1000 else ''}"
                
                # Fallback: show raw JSON
                return f"📋 Raw JSON Content:\n\n{json.dumps(lyrics_data, indent=2, ensure_ascii=False)[:1500]}{'...' if len(content) > 1500 else ''}"
                
            except json.JSONDecodeError:
                # If not JSON, show as plain text
                lines = content.split('\n')[:30]  # Show first 30 lines
                preview = "📝 Text Content:\n\n" + '\n'.join(lines)
                total_lines = len(content.split('\n'))
                if total_lines > 30:
                    remaining_lines = total_lines - 30
                    preview += f"\n\n... and {remaining_lines} more lines"
                return preview
                
        except Exception as e:
            return f"❌ Error reading file: {str(e)}"
        
    def parse_lyrics_file(self, lyrics_file):
        """Parse uploaded lyrics file"""
        if lyrics_file is None:
            return None
            
        try:
            # Handle both file objects and file paths
            if hasattr(lyrics_file, 'name'):
                file_path = lyrics_file.name
            else:
                file_path = lyrics_file
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            # Try to parse as JSON first
            try:
                lyrics_data = json.loads(content)
                
                # Handle different JSON formats
                if isinstance(lyrics_data, list) and len(lyrics_data) > 0:
                    # Standard format: [{"word": "hello", "start": 0.0, "end": 0.5}, ...]
                    if all(key in lyrics_data[0] for key in ['word', 'start', 'end']):
                        return lyrics_data
                
                elif isinstance(lyrics_data, dict):
                    # Complex format like the input files: {"char": [...], "word": [...]}
                    if 'word' in lyrics_data:
                        word_data = lyrics_data['word']
                        if isinstance(word_data, list) and len(word_data) > 0:
                            # Convert to standard format
                            processed_lyrics = []
                            for item in word_data:
                                if isinstance(item, dict) and all(key in item for key in ['word', 'start', 'end']):
                                    processed_lyrics.append({
                                        "word": item['word'],
                                        "start": item['start'], 
                                        "end": item['end']
                                    })
                            if processed_lyrics:
                                return processed_lyrics
                    
                    # Single word format
                    elif all(key in lyrics_data for key in ['word', 'start', 'end']):
                        return [lyrics_data]
                
                # If no recognized format, return the data as-is for the model to handle
                return lyrics_data
                
            except json.JSONDecodeError:
                # If not JSON, create simple timed lyrics from text
                lines = content.split('\n')
                lyrics_data = []
                current_time = 0.0
                
                for line in lines:
                    line = line.strip()
                    if line:
                        words = line.split()
                        for word in words:
                            lyrics_data.append({
                                "start": current_time,
                                "end": current_time + 0.5,
                                "word": word
                            })
                            current_time += 0.5
                
                return lyrics_data if lyrics_data else None
                
        except Exception as e:
            print(f"Error parsing lyrics file: {e}")
            return None
    
    def generate_style_embedding(self, audio_path, use_prompt_style=False, style_prompt=""):
        """Generate style embedding from audio or text prompt"""
        if use_prompt_style and style_prompt.strip():
            # Use text prompt for style
            style_prompt = style_prompt[:300]  # Limit length
            style_embedding = self.muq_model(texts=[style_prompt]).squeeze(0)
            return style_embedding
        else:
            # Use audio for style
            # Handle both file objects and file paths
            if hasattr(audio_path, 'name'):
                file_path = audio_path.name
            else:
                file_path = audio_path
                
            waveform, sample_rate = torchaudio.load(file_path)
            
            # Resample to 24kHz if needed
            if sample_rate != 24000:
                resampler = torchaudio.transforms.Resample(sample_rate, 24000)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            waveform = waveform.squeeze(0).to(self.device)
            
            # Crop to specified duration
            num_style_secs = self.config.evaluation.num_style_secs
            target_samples = 24000 * num_style_secs
            
            with torch.inference_mode():
                if waveform.shape[0] > target_samples:
                    waveform = waveform[:target_samples]
                style_embedding = self.muq_model(wavs=waveform.unsqueeze(0))
            
            return style_embedding[0]
    
    def generate_music(
        self, 
        reference_audio,
        lyrics_file, 
        style_prompt, 
        duration,
        use_prompt_style,
        cfg_strength,
        steps,
        ignore_style,
        progress=gr.Progress()
    ):
        """Main music generation function"""
        
        if reference_audio is None:
            return None, "❌ Please upload a reference audio file."
            
        if lyrics_file is None:
            return None, "❌ Please upload a lyrics file."
        
        try:
            progress(0.1, desc="Parsing lyrics...")
            
            # Parse lyrics
            lyrics_data = self.parse_lyrics_file(lyrics_file)
            if lyrics_data is None:
                return None, "❌ Failed to parse lyrics file. Please check the format."
            
            progress(0.2, desc="Generating style embedding...")
            
            # Generate style embedding
            style_embedding = self.generate_style_embedding(
                reference_audio, 
                use_prompt_style, 
                style_prompt
            )
            
            progress(0.4, desc="Preparing batch data...")
            
            # Create sample data
            sample_id = f"webui_{uuid.uuid4().hex[:8]}"
            
            # Calculate frames from duration
            frame_rate = 21.5
            num_frames = int(duration * frame_rate)
            fake_latent = torch.randn(128, num_frames)
            
            # Create fake sample tuple
            fake_sample = (
                sample_id,
                fake_latent,
                style_embedding,
                {'word': lyrics_data}
            )
            
            # Process through dataset
            processed_sample = self.base_dataset.process_sample_safely(fake_sample)
            
            if processed_sample is None:
                return None, "❌ Failed to process sample data."
            
            # Create batch
            batch = self.base_dataset.custom_collate_fn([processed_sample])
            
            if batch is None:
                return None, "❌ Failed to create batch."
            
            progress(0.6, desc="Generating audio latent...")
            
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
                elif isinstance(batch[key], list) and len(batch[key]) > 0 and isinstance(batch[key][0], torch.Tensor):
                    batch[key] = [item.to(self.device) for item in batch[key]]
            
            # Generate latent
            sample_kwargs = {
                "cfg_strength": cfg_strength,
                "steps": steps,
                "batch_infer_num": 1
            }
            
            latents = generate_latent(
                self.cfm_model, 
                batch, 
                sample_kwargs,
                negative_style_prompt_path=self.config.evaluation.negative_style_prompt,
                ignore_style=ignore_style
            )
            
            progress(0.8, desc="Decoding audio...")
            
            # Decode audio
            latent = latents[0][0]  # First sample, first inference
            latent_for_vae = latent.transpose(0, 1).unsqueeze(0)
            
            # Decode using VAE
            with torch.inference_mode():
                if self.config.evaluation.vae_type == 'diffrhythm':
                    pred_audio = self.vae.decode(latent_for_vae).sample
                else:
                    pred_audio = self.vae.decode_audio(latent_for_vae)
            
            # Normalize audio
            pred_audio = normalize_audio(pred_audio)
            
            progress(0.9, desc="Saving audio...")
            
            # Trim to requested duration
            sample_rate = 44100
            trim_samples = int(duration * sample_rate)
            if pred_audio.shape[-1] > trim_samples:  # Use -1 to handle any dimension order
                pred_audio = pred_audio[..., :trim_samples]
            
            # Ensure audio is 2D for torchaudio.save (channels, samples)
            if pred_audio.dim() == 1:
                pred_audio = pred_audio.unsqueeze(0)  # Add channel dimension
            elif pred_audio.dim() == 3:
                pred_audio = pred_audio.squeeze(0)   # Remove batch dimension
            
            # Save to temporary file
            temp_dir = tempfile.mkdtemp()
            output_path = os.path.join(temp_dir, f"{sample_id}.wav")
            torchaudio.save(output_path, pred_audio.cpu(), sample_rate)
            
            progress(1.0, desc="Generation complete!")
            
            generation_info = f"""✅ Generation completed successfully!

📊 Generation Details:
• Sample ID: {sample_id}
• Duration: {duration}s
• Style Source: {'Text Prompt' if use_prompt_style else 'Reference Audio'}
• CFG Strength: {cfg_strength}
• Generation Steps: {steps}
• Model: JAM-0.5
• Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🎵 Audio saved and ready for download!"""
            
            return output_path, generation_info
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Generation error: {error_details}")
            return None, f"❌ Generation failed: {str(e)}"

# Create Gradio interface
def create_interface(jam_webui):
    with gr.Blocks(
        title="JAM Music Generation WebUI", 
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {max-width: 1200px !important}
        .main-header {text-align: center; margin-bottom: 2rem;}
        .section-header {color: #2563eb; font-weight: bold;}
        """
    ) as demo:
        
        gr.HTML("""
        <div class="main-header">
            <h1>🎵 JAM Music Generation WebUI</h1>
            <p>Generate music using the JAM (Joint Audio and Lyrics) model</p>
            <p><strong>Status:</strong> Models loaded automatically on startup - Ready to generate!</p>
        </div>
        """)
        
        with gr.Group():
            gr.Markdown("### 📁 Input Files")
            
            reference_audio = gr.Audio(
                label="Reference Audio File",
                type="filepath"
            )
            gr.Markdown("*Upload an audio file to extract musical style from*")
            
            lyrics_file = gr.File(
                label="Lyrics File (JSON format)",
                file_types=[".json", ".txt"]
            )
            gr.Markdown("*Upload lyrics in JSON format with word timestamps*")
            
            # Lyrics preview area
            lyrics_preview = gr.Textbox(
                label="📖 Lyrics Preview",
                placeholder="Upload a lyrics file to see its content here...",
                lines=8,
                interactive=False,
                visible=True
            )
                    
        with gr.Accordion("📖 Lyrics Format Example", open=False):
            gr.Code("""[
    {"start": 0.0, "end": 0.5, "word": "Hello"},
    {"start": 0.5, "end": 1.0, "word": "world"},
    {"start": 1.0, "end": 1.5, "word": "this"},
    {"start": 1.5, "end": 2.0, "word": "is"},
    {"start": 2.0, "end": 2.5, "word": "music"}
]""", language="json")
        
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    gr.Markdown("### 🎨 Style Controls")
                    
                    use_prompt_style = gr.Checkbox(
                        label="Use Text Prompt for Style",
                        value=False
                    )
                    
                    style_prompt = gr.Textbox(
                        label="Style Prompt",
                        placeholder="Electronic dance music with heavy bass and synthesizers",
                        lines=2
                    )
                    gr.Markdown("*Only used if checkbox above is checked*")
                    
                    ignore_style = gr.Checkbox(
                        label="Ignore Style Conditioning",
                        value=False
                    )
                
            with gr.Column():
                with gr.Group():
                    gr.Markdown("### ⚙️ Generation Parameters")
                    
                    duration = gr.Slider(
                        label="Duration (seconds)",
                        minimum=10,
                        maximum=180,
                        value=30,
                        step=5
                    )
                    
                    cfg_strength = gr.Slider(
                        label="CFG Strength",
                        minimum=1.0,
                        maximum=10.0,
                        value=4.0,
                        step=0.1
                    )
                    
                    steps = gr.Slider(
                        label="Generation Steps",
                        minimum=10,
                        maximum=100,
                        value=50,
                        step=5
                    )
        
        with gr.Row():
            generate_btn = gr.Button("🎵 Generate Music", variant="primary", size="lg", scale=2)
        
        # Examples section
        gr.Markdown("### 🎯 Quick Examples")
        gr.Markdown("*Click on any example below to automatically fill in the input fields and try generation*")
        
        gr.Examples(
            examples=[
                [
                    "inputs/Hybrid Minds, Brodie - Heroin.mp3",
                    "inputs/Hybrid Minds, Brodie - Heroin.json",
                    "",  # style_prompt
                    30,  # duration
                    False,  # use_prompt_style
                    4.0,  # cfg_strength
                    50,  # steps
                    False  # ignore_style
                ],
                [
                    "inputs/Jade Bird - Avalanche.mp3",
                    "inputs/Jade Bird - Avalanche.json",
                    "",
                    30,
                    False,
                    4.0,
                    50,
                    False
                ],
                [
                    "inputs/Rizzle Kicks, Rachel Chinouriri - Follow Excitement!.mp3",
                    "inputs/Rizzle Kicks, Rachel Chinouriri - Follow Excitement!.json",
                    "",
                    35,
                    False,
                    4.5,
                    50,
                    False
                ],
                [
                    "inputs/Waylon Wyatt - Sincerely, Your Son.mp3",
                    "inputs/Waylon Wyatt - Sincerely, Your Son.json",
                    "Country folk ballad with acoustic guitar and emotional vocals",
                    40,
                    True,  # use text prompt style
                    3.5,
                    50,
                    False
                ]
            ],
            inputs=[
                reference_audio,
                lyrics_file,
                style_prompt,
                duration,
                use_prompt_style,
                cfg_strength,
                steps,
                ignore_style
            ],
            cache_examples=False,
            label="Sample Inputs - Click to Load"
        )
        
        with gr.Row():
            with gr.Column():
                output_audio = gr.Audio(
                    label="Generated Music",
                    type="filepath"
                )
                
            with gr.Column():
                generation_info = gr.Textbox(
                    label="Generation Info",
                    lines=12,
                    interactive=False
                )
        
        with gr.Accordion("💡 Usage Tips", open=False):
            gr.Markdown("""
            **Getting Started:**
            1. Upload a reference audio file (any music you like the style of)
            2. Upload a lyrics file in JSON format with word timings
            3. Adjust parameters as needed
            4. Click "Generate Music"
            
            **Tips for Better Results:**
            - **Duration**: Start with 30-60 seconds for faster generation
            - **CFG Strength**: 3-5 for balanced results, higher for stronger style adherence
            - **Steps**: 50 steps usually provides good quality
            - **Reference Audio**: Use high-quality audio for better style extraction
            - **Lyrics**: Make sure timestamps are accurate for better synchronization
            """)
        
        # Event handlers
        lyrics_file.change(
            fn=jam_webui.preview_lyrics,
            inputs=[lyrics_file],
            outputs=[lyrics_preview]
        )
        
        generate_btn.click(
            fn=jam_webui.generate_music,
            inputs=[
                reference_audio,
                lyrics_file,
                style_prompt,
                duration,
                use_prompt_style,
                cfg_strength,
                steps,
                ignore_style
            ],
            outputs=[output_audio, generation_info],
            show_progress=True
        )
    
    return demo

if __name__ == "__main__":
    print("🚀 Starting JAM Music Generation WebUI...")
    
    # Create global instance
    jam_webui = JAMWebUI()
    
    demo = create_interface(jam_webui)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
