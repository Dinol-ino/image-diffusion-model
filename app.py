import tkinter as tk
import customtkinter as ctk
from PIL import ImageTk
import threading
import torch
from diffusers import StableDiffusionPipeline


app = tk.Tk()
app.geometry("532x622")
app.title("Stable Bud")
ctk.set_appearance_mode("dark")

prompt = ctk.CTkEntry(app, width=512, height=40)
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(app, width=512, height=512, text="")
lmain.place(x=10, y=60)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = None

def load_model():
    global pipe
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        use_auth_token=hf_read
    )
    pipe.to("cpu")
    pipe.enable_attention_slicing()


threading.Thread(target=load_model, daemon=True).start()

def generate():
    if pipe is None:
        print("Model not loaded yet")
        return

    def task():
        with torch.autocast(device_type=device, enabled=(device=="cuda")):
            image = pipe(prompt.get(), guidance_scale=8.5).images[0]
            image.save("generated.png")

            img = ImageTk.PhotoImage(image)
            lmain.configure(image=img)
            lmain.image = img

    threading.Thread(target=task, daemon=True).start()

trigger = ctk.CTkButton(app, text="Generate", command=generate)
trigger.place(x=206, y=580)

app.mainloop()
