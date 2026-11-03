import zipfile
import os

zip_path = "ACTIVEA_Codebase_Siva_Subramaniam_Raju_260322784.zip"
print(f"Archive Name: {zip_path}")
print(f"Archive Size: {os.path.getsize(zip_path) / (1024*1024):.2f} MB")

with zipfile.ZipFile(zip_path, "r") as z:
    infolist = z.infolist()
    print(f"Total Items in Archive: {len(infolist)}")
    
    bad_files = []
    extensions = {}
    total_uncompressed = 0
    
    for info in infolist:
        total_uncompressed += info.file_size
        ext = os.path.splitext(info.filename)[1].lower()
        extensions[ext] = extensions.get(ext, 0) + 1
        
        # Check for bad patterns
        for bad in ["__pycache__", ".pyc", ".git", ".cursor", "scratch", "tmp", "endomondoHR.json", "endomondoHR.csv", ".docx", ".pdf", ".pptx"]:
            if bad in info.filename:
                bad_files.append(info.filename)
                
    print("\nFile Type Breakdown:")
    for ext, count in sorted(extensions.items()):
        name = ext if ext else "[directory/other]"
        print(f"  {name:<15} : {count} files")
        
    print(f"\nTotal Uncompressed Size: {total_uncompressed / (1024*1024):.2f} MB")
    
    if bad_files:
        print(f"\nWARNING: Found {len(bad_files)} forbidden/unwanted files:")
        for b in bad_files:
            print(f"   ! {b}")
    else:
        print("\nCHECK PASSED: Zero unwanted files (No pycache, no docx/pptx, no 6GB raw data)!")
