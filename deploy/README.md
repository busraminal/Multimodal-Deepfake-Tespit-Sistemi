# Hetzner / Linux uzerinde calistirma

## Gereksinimler

- Sunucuda Docker + Docker Compose plugin
- Gorsel model agirligi: `models/faceforensics/full/full_c23.p` (veya `DF_VISUAL_MODEL_PATH` ile baska yol). Dosya yoksa uygulama goruntu tarafinda heuristik moda duser.

## Hizli baslat

Repoyu sunucuya klonla, model dosyasini `models/` altina koy, sonra:

```bash
docker compose up -d --build
```

Arayuz: `http://SUNUCU_IP:8501`

## TLS (onerilen)

8501 portunu disari acmak yerine ters proxy kullan:

- **Caddy** veya **Nginx** ile Let's Encrypt
- Ornek: Caddyfile ile `reverse_proxy localhost:8501`

Boylece HTTPS ve kolay alan adi elde edersin.

## Ortam degiskenleri

| Degisken | Aciklama |
|----------|-----------|
| `FFMPEG_PATH` / `FFMPEG_BIN` | ffmpeg tam yolu (opsiyonel; yoksa PATH) |
| `FFMPEG_QUIET` | `1` ise ffmpeg-python sessiz (Docker varsayilan) |
| `DF_VISUAL_MODEL_PATH` | Xception checkpoint yolu |

## GPU (opsiyonel)

Bu `Dockerfile` CPU PyTorch kullanir. GPU sunucuda `Dockerfile` icindeki torch kurulum satirini CUDA indeksine gore degistirmen gerekir.

## Egitim (ayri is)

Buyuk veri seti icin egitimi ayri bir makinede veya `docker compose run --rm app python train/...` ile calistirabilirsin; uretim konteynerinde sadece inference tutmak daha temiz olur.
