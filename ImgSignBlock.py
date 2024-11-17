# ImgSign.py
import argparse
import json
from image_watermarking import ImageSigner, ImageVerifier

def print_correlation_grid(correlation_grid):
    """Print correlation grid with ASCII representation"""
    print("\nBlock Correlation Grid:")
    print("-" * 41)
    
    for row in correlation_grid:
        value_row = " ".join(f"{x:5.2f}" for x in row)
        print(f"| {value_row}   |")
        #this should be adjusted in case you modify treshold in the backedn (legend as well)
        ascii_row = "  "
        for corr in row:
            if corr > 0.6:
                ascii_row += "██    "  #TODO - Improve vizualisation formating
            elif corr > 0.4:
                ascii_row += "▓▓    "
            else:
                ascii_row += "░░    "
        print(f"| {ascii_row}|")
        print("-" * 41)
    
    print("\nLegend:")
    print("██ - Authentic (correlation > 0.6)")
    print("▓▓ - Suspicious (correlation 0.4-0.6)")
    print("░░ - Modified (correlation < 0.4)")

def sign_image(input_path, output_path, secret_key):
    """Sign an image and return its signature"""
    try:
        signer = ImageSigner(secret_key=secret_key)
        signature = signer.embed_watermark(input_path, output_path)
        
        signature_file = output_path + '.sig'
        with open(signature_file, 'w') as f:
            f.write(signature)
            
        return signature
    except Exception as e:
        raise Exception(f"Error signing image: {str(e)}")

def verify_image(image_path, signature_path, secret_key):
    """Verify if an image contains the expected watermark"""
    try:
        verifier = ImageVerifier(secret_key=secret_key)
        
        with open(signature_path, 'r') as f:
            signature = f.read()
            
        result = verifier.verify_signature(image_path, signature)
        return result
    except Exception as e:
        raise Exception(f"Error verifying image: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Image Block Signing and Verification Tool')
    parser.add_argument('action', choices=['sign', 'verify'], 
                       help='Action to perform: sign or verify the image')
    parser.add_argument('-i', '--input', required=True,
                       help='Path to image')
    parser.add_argument('-o', '--output',
                       help='Output path for signed image (required for sign action)')
    parser.add_argument('-k', '--key', required=True,
                       help='Secret key for signing/verification, string not file')
    parser.add_argument('-s', '--signature',
                       help='Path to signature file for img verification (required for verify action)')

    args = parser.parse_args()

    try:
        if args.action == 'sign':
            if not args.output:
                parser.error("--output path is required for image signing")
                
            signature = sign_image(
                input_path=args.input,
                output_path=args.output,
                secret_key=args.key
            )
            print("\nImage Signing Results:")
            print(f"Input image: {args.input}")
            print(f"Watermarked image saved as: {args.output}")
            print(f"Signature saved as: {args.output}.sig")

        elif args.action == 'verify':
            if not args.signature:
                parser.error("--signature is required for image verification")
                
            result = verify_image(
                image_path=args.input,
                signature_path=args.signature,
                secret_key=args.key
            )
            
            print("\nVerification Results:")
            print(f"Image being verified: {args.input}")
            print(f"Signature valid: {result['signature_valid']}")
            print(f"Status: {result['status']}")
            
            stats = result['stats']
            print(f"\nStatistics:")
            print(f"Average correlation: {stats['average_correlation']:.3f}")
            print(f"Standard deviation: {stats['std_correlation']:.3f}")
            print(f"Valid blocks: {stats['valid_blocks']}/{stats['total_blocks']}")
            
            print_correlation_grid(result['correlation_grid'])

    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()