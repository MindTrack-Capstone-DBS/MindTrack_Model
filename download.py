# download.py
import nltk

print("--> Memulai proses unduh data NLTK...")
try:
    nltk.download('punkt', quiet=False)  # quiet=False agar lebih vokal
    print("--> Unduhan 'punkt' berhasil.")
    nltk.download('stopwords', quiet=False)
    print("--> Unduhan 'stopwords' berhasil.")
    nltk.download('wordnet', quiet=False)
    print("--> Unduhan 'wordnet' berhasil.")
    print("--> Semua data NLTK siap.")
except Exception as e:
    print(f"--> Terjadi error saat mengunduh data: {e}")
    # Keluar dengan status error agar proses deploy berhenti
    exit(1)
