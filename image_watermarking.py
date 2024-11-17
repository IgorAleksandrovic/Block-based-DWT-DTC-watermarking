import numpy as np
from PIL import Image
import pywt
import cv2
import hashlib
import hmac
import base64
from scipy.fftpack import dct, idct

class WatermarkBase:
    """Base class containing common functionality for signing and verifying watermarks"""
    def __init__(self, secret_key):
        # Core parameters
        self.secret_key = secret_key.encode('utf-8')
        self.wavelet = 'db1'
        self.level = 2
        self.block_size = 8
        self.grid_size = 4
        
        # Processing parameters
        self.max_dimension = 2048  # Maximum image dimension
        self.strength = 3          # Watermark embedding strength (Find a sweet spot between visibility and robustness)
        self.threshold = 0.3       # Correlation threshold for verification (Lower value is more robust but less sensitive)
                                   # If you change treshold, dajust display in ImgSignBlock
                                    
        # Luminance weights for adaptive watermark aplication
        self.luminance_scale = 128.0  # Used for normalisation, divide mean luminance by this value
        self.luminance_range = (0.3, 1.0)  # Min/max clip values (to set bounds in case of very dark or bright blocks)
        
        # Texture weight for adaptive watermark aplication
        self.texture_scale = 30.0  # Used for normalisation, divide mean texture by this value
        self.texture_base = 1.0  # Base value for texture weight, increase for strength and decrese for visibility
        self.texture_range = (0.5, 2.0)  # Min/max clip values (to set bounds in case of very detailed of flat blocks)
        
        # Sub-band weights (Horizontal, Vertical, Diagonal)
        # This determines signature strength in each of the DWT bands
        self.sub_band_weights = (1.0, 1.2, 1.2)
        
        
    def calculate_luminance_weight(self, mean_luminance):
        return np.clip(
            mean_luminance / self.luminance_scale,
            self.luminance_range[0],
            self.luminance_range[1]
        )    
        
    def calculate_texture_weight(self, texture):
        return np.clip(
            self.texture_base + np.mean(texture) / self.texture_scale,
            self.texture_range[0],
            self.texture_range[1]
        )

    def get_sub_band_weight(self, band_index):
        return self.sub_band_weights[band_index]

    def normalize_image(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)
            
        height, width = img.shape[:2]
        aspect_ratio = width / height
        
        if width > height:
            new_width = self.max_dimension
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = self.max_dimension
            new_width = int(new_height * aspect_ratio)
            
        new_width = (new_width // 2) * 2
        new_height = (new_height // 2) * 2
        
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        return resized

    def generate_block_watermark(self, shape, block_index):
        """Unique for each block"""
        block_data = f"{self.secret_key.decode('utf-8')}:block{block_index}".encode('utf-8')
        block_key = hashlib.sha256(block_data).digest()
        np.random.seed(int.from_bytes(block_key[:4], 'big'))
        watermark = np.random.randn(*shape)
        return watermark / np.sqrt(np.mean(watermark**2))

    def split_coeffs_into_blocks(self, coeffs):
        """Split DWT coefs of a single band into blocks"""
        h, w = coeffs.shape
        pad_h = (self.grid_size - h % self.grid_size) % self.grid_size
        pad_w = (self.grid_size - w % self.grid_size) % self.grid_size
        
        if pad_h > 0 or pad_w > 0:
            coeffs = np.pad(coeffs, ((0, pad_h), (0, pad_w)), mode='reflect')
            
        h, w = coeffs.shape
        block_h, block_w = h // self.grid_size, w // self.grid_size
        blocks = []
        positions = []
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                start_h = i * block_h
                start_w = j * block_w
                block = coeffs[start_h:start_h + block_h, start_w:start_w + block_w]
                blocks.append(block)
                positions.append((start_h, start_w, block_h, block_w))
                
        return blocks, positions, (pad_h, pad_w)

    def _generate_signature(self, watermarks):
        h = hmac.new(self.secret_key, digestmod=hashlib.sha256)
        for watermark in watermarks:
            h.update(watermark.tobytes())
        return base64.b64encode(h.digest()).decode('utf-8')


class ImageSigner(WatermarkBase):
    def __init__(self, secret_key):
        super().__init__(secret_key=secret_key)
        
    def calculate_mask(self, block):
        """Based on HVS - basically high frequency and luminance areas can support more changes before human noticing"""
        mean_luminance = np.mean(block)
        luminance_weight = self.calculate_luminance_weight(mean_luminance)
        
        texture = np.abs(block - cv2.GaussianBlur(block, (3,3), 0))
        texture_weight = self.calculate_texture_weight(texture)
        
        return luminance_weight * texture_weight

    def process_dct_block(self, block, watermark):
        """For a single block"""
        height, width = block.shape
        dct_pad_h = (self.block_size - height % self.block_size) % self.block_size
        dct_pad_w = (self.block_size - width % self.block_size) % self.block_size
        
        if dct_pad_h > 0 or dct_pad_w > 0:
            padded_block = np.pad(block, ((0, dct_pad_h), (0, dct_pad_w)), mode='reflect')
        else:
            padded_block = block

        if watermark.shape != padded_block.shape:
            watermark = cv2.resize(watermark, (padded_block.shape[1], padded_block.shape[0]))
            
        perceptual_mask = self.calculate_mask(padded_block)
        
        # DCT
        dct_coeffs = np.zeros_like(padded_block)
        for i in range(0, padded_block.shape[0], self.block_size):
            for j in range(0, padded_block.shape[1], self.block_size):
                b = padded_block[i:i+self.block_size, j:j+self.block_size]
                dct_coeffs[i:i+self.block_size, j:j+self.block_size] = dct(dct(b.T, norm='ortho').T, norm='ortho')
        
        for i in range(0, dct_coeffs.shape[0], self.block_size):
            for j in range(0, dct_coeffs.shape[1], self.block_size):
                 # To target (mostly) mid freq in the block
                mid_freq = dct_coeffs[i+3:i+5, j+3:j+5]
                mid_energy = np.sqrt(np.mean(mid_freq**2))
                if mid_energy > 0:
                    dct_coeffs[i+3:i+5, j+3:j+5] += (
                        mid_energy * self.strength * 
                        watermark[i+3:i+5, j+3:j+5] * perceptual_mask
                    )
        
        idct_coeffs = np.zeros_like(dct_coeffs) #Pack it back up with inverse transform
        for i in range(0, dct_coeffs.shape[0], self.block_size):
            for j in range(0, dct_coeffs.shape[1], self.block_size):
                b = dct_coeffs[i:i+self.block_size, j:j+self.block_size]
                idct_coeffs[i:i+self.block_size, j:j+self.block_size] = idct(idct(b.T, norm='ortho').T, norm='ortho')
        
        if dct_pad_h > 0 or dct_pad_w > 0:
            return idct_coeffs[:height, :width]
        return idct_coeffs

    def embed_watermark(self, image_path, output_path):
        img = self.normalize_image(np.array(Image.open(image_path)))
        
        if len(img.shape) == 3:  #we are using the luminance (Y) channel, if grayscale just use floating point values
            img_yuv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2YUV)
            y_channel = img_yuv[:,:,0].astype(float)
        else:
            y_channel = img.astype(float)

        coeffs = pywt.wavedec2(y_channel, wavelet=self.wavelet, level=self.level)
        coeffs_modified = list(coeffs)
        detail_coeffs = list(coeffs_modified[1])
        watermarks = []
        
        
        for sub_band_idx in range(3): #split band into grid...
            sub_band = detail_coeffs[sub_band_idx]
            original_shape = sub_band.shape
            
            blocks, positions, grid_padding = self.split_coeffs_into_blocks(sub_band)
            
            processed_blocks = []
            for block_idx, block in enumerate(blocks):
                watermark = self.generate_block_watermark(block.shape, block_idx)
                watermarks.append(watermark)
                processed_block = self.process_dct_block(
                    block, 
                    watermark * self.get_sub_band_weight(sub_band_idx)
                )
                processed_blocks.append(processed_block)
            
            h, w = sub_band.shape #...and put it back togeather
            if grid_padding[0] > 0 or grid_padding[1] > 0:
                h += grid_padding[0]
                w += grid_padding[1]
            
            processed_band = np.zeros((h, w))
            
            for (block, pos) in zip(processed_blocks, positions):
                start_h, start_w, bh, bw = pos
                processed_band[start_h:start_h + bh, start_w:start_w + bw] = block
            
            if grid_padding[0] > 0 or grid_padding[1] > 0: #need to remove padding
                processed_band = processed_band[:original_shape[0], :original_shape[1]]
            
            detail_coeffs[sub_band_idx] = processed_band
        
        coeffs_modified[1] = tuple(detail_coeffs)
        
        watermarked = pywt.waverec2(coeffs_modified, wavelet=self.wavelet)
        watermarked = np.clip(watermarked, 0, 255)

        if len(img.shape) == 3:
            img_yuv[:,:,0] = watermarked
            watermarked_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        else:
            watermarked_img = watermarked
            
        Image.fromarray(np.uint8(watermarked_img)).save(output_path, quality=95)
        return self._generate_signature(watermarks)

class ImageVerifier(WatermarkBase):
    def __init__(self, secret_key):
        super().__init__(secret_key=secret_key)

    def calculate_block_correlation(self, block, watermark):
        height, width = block.shape
        dct_pad_h = (self.block_size - height % self.block_size) % self.block_size
        dct_pad_w = (self.block_size - width % self.block_size) % self.block_size
        
        if dct_pad_h > 0 or dct_pad_w > 0:
            padded_block = np.pad(block, ((0, dct_pad_h), (0, dct_pad_w)), mode='reflect')
        else:
            padded_block = block

        if watermark.shape != padded_block.shape:
            watermark = cv2.resize(watermark, (padded_block.shape[1], padded_block.shape[0]))

        # DCT
        dct_coeffs = np.zeros_like(padded_block)
        for i in range(0, padded_block.shape[0], self.block_size):
            for j in range(0, padded_block.shape[1], self.block_size):
                b = padded_block[i:i+self.block_size, j:j+self.block_size]
                dct_coeffs[i:i+self.block_size, j:j+self.block_size] = dct(dct(b.T, norm='ortho').T, norm='ortho')

        correlations = []
        for i in range(0, dct_coeffs.shape[0], self.block_size):
            for j in range(0, dct_coeffs.shape[1], self.block_size):
                # Only check mid frequencies
                mid_corr = np.corrcoef(
                    dct_coeffs[i+3:i+5, j+3:j+5].flatten(),
                    watermark[i+3:i+5, j+3:j+5].flatten()
                )[0,1]
                
                correlations.append(0 if np.isnan(mid_corr) else mid_corr)
        
        return np.mean(correlations)

    def verify_signature(self, image_path, signature):
        img = np.array(Image.open(image_path))
        normalized_img = self.normalize_image(img)
        
        if len(normalized_img.shape) == 3:
            img_yuv = cv2.cvtColor(normalized_img.astype(np.uint8), cv2.COLOR_RGB2YUV)
            y_channel = img_yuv[:,:,0].astype(float)
        else:
            y_channel = normalized_img.astype(float)

        # DWT
        coeffs = pywt.wavedec2(y_channel, wavelet=self.wavelet, level=self.level)
        detail_coeffs = coeffs[1]
        
        watermarks = []
        all_sub_band_correlations = []
        
        
        # Process sub-bands
        for sub_band_idx in range(3):
            sub_band = detail_coeffs[sub_band_idx]
            
            blocks, positions, grid_padding = self.split_coeffs_into_blocks(sub_band)
            
            sub_band_correlations = []
            for block_idx, block in enumerate(blocks):
                watermark = self.generate_block_watermark(block.shape, block_idx)
                watermarks.append(watermark)
                
                correlation = self.calculate_block_correlation(
                    block, 
                    watermark * self.get_sub_band_weight(sub_band_idx)
                )
                sub_band_correlations.append(correlation)
            
            all_sub_band_correlations.append(sub_band_correlations)
        
        # Average the correlations across sub-bands
        block_correlations = np.mean(all_sub_band_correlations, axis=0)
        
        # Calculate statistics
        correlations = np.array(block_correlations)
        avg_correlation = np.mean(correlations)
        std_correlation = np.std(correlations)
        valid_blocks = np.sum(correlations > self.threshold)
        
        # Modify this to fine tune
        if avg_correlation < 0.2:
            status = "UNSIGNED"
        elif std_correlation > 0.15:
            status = "MODIFIED"
        else:
            status = "AUTHENTIC"
        
        expected_signature = self._generate_signature(watermarks)
        signature_valid = hmac.compare_digest(signature.encode('utf-8'), 
                                            expected_signature.encode('utf-8'))
        
        # Create grid from correlations for pretty display
        correlation_grid = correlations.reshape(self.grid_size, self.grid_size)
        
        return {
            'signature_valid': signature_valid,
            'status': status,
            'correlation_grid': correlation_grid,
            'stats': {
                'average_correlation': avg_correlation,
                'std_correlation': std_correlation,
                'valid_blocks': valid_blocks,
                'total_blocks': len(block_correlations)
            }
        }