import json
import webbrowser
import socket
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import time
import os
from pathlib import Path
import urllib.parse


# Global reference to QC manager
_QC_MANAGER = None


class QCHandler(SimpleHTTPRequestHandler):
    
    def do_GET(self):
        try:
            # Parse the path
            path = self.path.split('?')[0]  # Remove query parameters
            path = path.lstrip('/')
            
            print(f"\n{'='*60}")
            print(f"📥 GET Request:")
            print(f"   Raw path: {self.path}")
            print(f"   Cleaned path: '{path}'")
            
            # Decode URL encoding
            decoded_path = urllib.parse.unquote(path)
            print(f"   Decoded path: '{decoded_path}'")
            
            # Get the current directory
            current_dir = os.getcwd()
            print(f"   Current directory: {current_dir}")
            
            # Check if this is an image request
            if path.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
                # Try different variations
                variations = [
                    path,  # Original
                    decoded_path,  # Decoded
                    path.replace('%20', ' '),  # Replace %20 with space
                    path.replace('%28', '(').replace('%29', ')'),  # Replace parentheses
                    decoded_path.replace('%20', ' '),  # Decoded with spaces
                ]
                
                # Remove duplicates
                variations = list(dict.fromkeys(variations))
                
                print(f"\nLooking for image file:")
                for variant in variations:
                    file_path = os.path.join(current_dir, variant)
                    exists = os.path.exists(file_path)
                    print(f"   '{variant}' -> exists: {exists}")
                    if exists:
                        print(f"FOUND at: {file_path}")
                        # Serve the file
                        self.path = '/' + variant
                        return SimpleHTTPRequestHandler.do_GET(self)
                
                # If we get here, file wasn't found
                print(f"File not found in any variation")
                
                # List all files in directory that might match
                print(f"\nFiles in directory (first 20):")
                all_files = os.listdir(current_dir)
                jpg_files = [f for f in all_files if f.endswith(('.jpg', '.jpeg', '.png'))]
                for f in jpg_files[:20]:
                    print(f"   {f}")
                
                # Look for similar names
                print(f"\nLooking for similar files:")
                # Extract base name from path (remove _centers.jpg)
                if '_centers.' in path:
                    base = path.split('_centers.')[0]
                elif '_centers' in path:
                    base = path.split('_centers')[0]
                else:
                    base = path.split('.')[0]
                
                print(f"Searching for files containing: '{base}'")
                for f in jpg_files:
                    if base in f or base.replace(' ', '') in f.replace(' ', ''):
                        print(f"Found similar: {f}")
                
                # If we still can't find it, serve a placeholder
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                placeholder = f"""
                <html>
                <body style="background:#f0f0f0; display:flex; justify-content:center; align-items:center; height:200px; margin:0;">
                    <div style="text-align:center; color:#999; font-family:Arial;">
                        <h2>Image Not Found</h2>
                        <p>Looking for: {path}</p>
                        <p style="font-size:12px;">Decoded: {decoded_path}</p>
                    </div>
                </body>
                </html>
                """
                self.wfile.write(placeholder.encode('utf-8'))
                return
            
            # For non-image requests, serve normally
            return SimpleHTTPRequestHandler.do_GET(self)
            
        except Exception as e:
            print(f"Error in GET: {e}")
            import traceback
            traceback.print_exc()
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f"Error: {e}".encode())
    
    def do_POST(self):

        try:
            # Only handle /delete endpoint
            if self.path != '/delete':
                self.send_response(404)
                self.end_headers()
                return
            
            # Read the request body
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self._send_json(400, {'success': False, 'message': 'No data provided'})
                return
            
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            image_name = data.get('image_name')
            
            print(f"\nDelete request for: {image_name}")
            
            if not image_name:
                self._send_json(400, {'success': False, 'message': 'No image name provided'})
                return
            
            # Use the global QC manager
            if _QC_MANAGER is None:
                print("QC Manager is None!")
                self._send_json(500, {'success': False, 'message': 'QC Manager not initialized'})
                return
            
            # Delete the image
            try:
                success = _QC_MANAGER._delete_image(image_name)
                print(f"Delete result: {success}")
            except Exception as e:
                print(f"Error in _delete_image: {e}")
                import traceback
                traceback.print_exc()
                self._send_json(500, {'success': False, 'message': f'Delete error: {str(e)}'})
                return
            
            if success:
                self._send_json(200, {'success': True, 'message': f'Deleted {image_name}'})
            else:
                self._send_json(500, {'success': False, 'message': f'Failed to delete {image_name}'})
                
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            self._send_json(400, {'success': False, 'message': f'Invalid JSON: {e}'})
        except Exception as e:
            print(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            self._send_json(500, {'success': False, 'message': str(e)})
    
    def _send_json(self, status_code, data):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def log_message(self, format, *args):
        pass


def start_qc_server(qc_manager, port=8888):

    global _QC_MANAGER
    _QC_MANAGER = qc_manager
    
    print(f"QC Manager set in server")
    print(f"QC folder: {qc_manager.qc_folder}")
    
    # Find an available port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    sock.close()
    
    if result == 0:
        port += 1
        print(f"Port {port-1} in use, using port {port}")
    
    # Create server
    server = HTTPServer(('127.0.0.1', port), QCHandler)
    
    # Open browser
    url = f"http://127.0.0.1:{port}/qc_report.html"
    
    def open_browser():
        time.sleep(1.5)
        webbrowser.open(url)
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    print(f"URL: {url}")
    print(f"\nClick 'Delete' on any image to remove it.")
    print(f"Press Ctrl+C when done.")
    
    return server


def stop_qc_server(server):

    try:
        server.shutdown()
        server.server_close()
        print("\nQC server stopped.")
    except:
        pass