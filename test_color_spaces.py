import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def test_color_space(image_path1, image_path2, color_space='hsv', channel=None, output_dir='color_tests'):
    if not os.path.exists(image_path1) or not os.path.exists(image_path2):
        print("One or both images not found")
        return None
    
    # Read images
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    
    # Convert BGR to RGB for display
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    if color_space == 'hsv':
        # HSV: Hue, Saturation, Value
        converted1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        converted2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        
        h1, s1, v1 = cv2.split(converted1)
        h2, s2, v2 = cv2.split(converted2)
        
        channels = {'hue': (h1, h2), 'saturation': (s1, s2), 'value': (v1, v2)}
        channel_names = {'hue': 'Hue (0-179)', 'saturation': 'Saturation', 'value': 'Value (Brightness)'}
        
        if channel == 'all' or channel is None:
            for name, (ch1, ch2) in channels.items():
                results[name] = compare_channels(img1_rgb, img2_rgb, ch1, ch2, 
                                                  f"{color_space}_{name}", output_dir, 
                                                  channel_names.get(name, name))
        else:
            if channel in channels:
                ch1, ch2 = channels[channel]
                results[channel] = compare_channels(img1_rgb, img2_rgb, ch1, ch2,
                                                     f"{color_space}_{channel}", output_dir,
                                                     channel_names.get(channel, channel))
    
    elif color_space == 'lab':
        # LAB: Luminance, A (green-red), B (blue-yellow)
        converted1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
        converted2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
        
        l1, a1, b1 = cv2.split(converted1)
        l2, a2, b2 = cv2.split(converted2)
        
        channels = {'luminance': (l1, l2), 'a_channel': (a1, a2), 'b_channel': (b1, b2)}
        
        if channel == 'all' or channel is None:
            for name, (ch1, ch2) in channels.items():
                results[name] = compare_channels(img1_rgb, img2_rgb, ch1, ch2,
                                                  f"{color_space}_{name}", output_dir, name)
        else:
            ch1, ch2 = channels.get(channel, (None, None))
            if ch1 is not None:
                results[channel] = compare_channels(img1_rgb, img2_rgb, ch1, ch2,
                                                     f"{color_space}_{channel}", output_dir, channel)
    
    elif color_space == 'yuv':
        # YUV: Luma, Chrominance
        converted1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YUV)
        converted2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YUV)
        
        y1, u1, v1 = cv2.split(converted1)
        y2, u2, v2 = cv2.split(converted2)
        
        channels = {'y_luma': (y1, y2), 'u_chroma': (u1, u2), 'v_chroma': (v1, v2)}
        
        if channel == 'all' or channel is None:
            for name, (ch1, ch2) in channels.items():
                results[name] = compare_channels(img1_rgb, img2_rgb, ch1, ch2,
                                                  f"{color_space}_{name}", output_dir, name)
        else:
            ch1, ch2 = channels.get(channel, (None, None))
            if ch1 is not None:
                results[channel] = compare_channels(img1_rgb, img2_rgb, ch1, ch2,
                                                     f"{color_space}_{channel}", output_dir, channel)
    
    elif color_space == 'hls':
        # HLS: Hue, Lightness, Saturation
        converted1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HLS)
        converted2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HLS)
        
        h1, l1, s1 = cv2.split(converted1)
        h2, l2, s2 = cv2.split(converted2)
        
        channels = {'hue': (h1, h2), 'lightness': (l1, l2), 'saturation': (s1, s2)}
        
        if channel == 'all' or channel is None:
            for name, (ch1, ch2) in channels.items():
                results[name] = compare_channels(img1_rgb, img2_rgb, ch1, ch2,
                                                  f"{color_space}_{name}", output_dir, name)
        else:
            ch1, ch2 = channels.get(channel, (None, None))
            if ch1 is not None:
                results[channel] = compare_channels(img1_rgb, img2_rgb, ch1, ch2,
                                                     f"{color_space}_{channel}", output_dir, channel)
    
    elif color_space == 'gray':
        # Grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        results['grayscale'] = compare_channels(img1_rgb, img2_rgb, gray1, gray2,
                                                 f"{color_space}", output_dir, 'Grayscale')
    
    elif color_space == 'rgb':
        # RGB individual channels
        r1, g1, b1 = cv2.split(img1)
        r2, g2, b2 = cv2.split(img2)
        
        channels = {'red': (r1, r2), 'green': (g1, g2), 'blue': (b1, b2)}
        
        if channel == 'all' or channel is None:
            for name, (ch1, ch2) in channels.items():
                results[name] = compare_channels(img1_rgb, img2_rgb, ch1, ch2,
                                                  f"{color_space}_{name}", output_dir, name)
        else:
            ch1, ch2 = channels.get(channel, (None, None))
            if ch1 is not None:
                results[channel] = compare_channels(img1_rgb, img2_rgb, ch1, ch2,
                                                     f"{color_space}_{channel}", output_dir, channel)
    
    return results

def compare_channels(img1_rgb, img2_rgb, channel1, channel2, name, output_dir, title_prefix):

    # Normalize for display
    ch1_norm = cv2.normalize(channel1, None, 0, 255, cv2.NORM_MINMAX)
    ch2_norm = cv2.normalize(channel2, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply colormap for better visualization
    ch1_color = cv2.applyColorMap(ch1_norm, cv2.COLORMAP_JET)
    ch2_color = cv2.applyColorMap(ch2_norm, cv2.COLORMAP_JET)
    
    # Create side-by-side comparison
    h1, w1 = img1_rgb.shape[:2]
    h2, w2 = img2_rgb.shape[:2]
    
    # Resize to same height for display
    target_h = min(h1, h2)
    if h1 != target_h:
        ch1_color = cv2.resize(ch1_color, (int(w1 * target_h / h1), target_h))
    if h2 != target_h:
        ch2_color = cv2.resize(ch2_color, (int(w2 * target_h / h2), target_h))
    
    comparison = np.hstack([ch1_color, ch2_color])
    
    # Add labels
    label_height = 30
    labeled = np.vstack([
        np.zeros((label_height, comparison.shape[1], 3), dtype=np.uint8),
        comparison
    ])
    cv2.putText(labeled, f"Image 1 - {title_prefix}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(labeled, f"Image 2 - {title_prefix}", (comparison.shape[1]//2 + 10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    output_path = os.path.join(output_dir, f"{name}_comparison.jpg")
    cv2.imwrite(output_path, labeled)
    
    # Calculate statistics
    mean1 = np.mean(channel1)
    mean2 = np.mean(channel2)
    std1 = np.std(channel1)
    std2 = np.std(channel2)
    
    print(f"\n📊 {title_prefix.upper()}:")
    print(f"   Image 1 - Mean: {mean1:.2f}, Std: {std1:.2f}")
    print(f"   Image 2 - Mean: {mean2:.2f}, Std: {std2:.2f}")
    print(f"   Difference - Mean: {abs(mean1 - mean2):.2f}")
    
    return {'mean1': mean1, 'mean2': mean2, 'std1': std1, 'std2': std2}


def visualize_original_comparison(image_path1, image_path2, output_dir='color_tests'):
    
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Resize to same height
    h1, w1 = img1_rgb.shape[:2]
    h2, w2 = img2_rgb.shape[:2]
    target_h = min(h1, h2)
    
    if h1 != target_h:
        img1_rgb = cv2.resize(img1_rgb, (int(w1 * target_h / h1), target_h))
    if h2 != target_h:
        img2_rgb = cv2.resize(img2_rgb, (int(w2 * target_h / h2), target_h))
    
    comparison = np.hstack([img1_rgb, img2_rgb])
    
    # Add labels
    label_height = 30
    labeled = np.vstack([
        np.zeros((label_height, comparison.shape[1], 3), dtype=np.uint8),
        comparison
    ])
    cv2.putText(labeled, "Image 1 - Original", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(labeled, "Image 2 - Original", (comparison.shape[1]//2 + 10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "original_comparison.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(labeled, cv2.COLOR_RGB2BGR))
    
    print(f"\n📷 Original images saved to: {output_path}")
    
    # Display
    plt.figure(figsize=(12, 6))
    plt.imshow(labeled)
    plt.axis('off')
    plt.title('Original Images Comparison')
    plt.tight_layout()
    plt.show()
