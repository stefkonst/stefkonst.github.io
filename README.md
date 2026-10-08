# Personal faculty-application site

Static site, no build step. Everything lives in this folder:

```
site/
  index.html
  assets/
    img/    optimized photos & figures
    video/  compressed video clips
    docs/   CV PDF (and any statement PDFs you add)
```

## Deploy to GitHub Pages

From inside this `site/` folder:

```bash
git init
git add .
git commit -m "Initial site"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Then on GitHub: **Settings → Pages → Source → Deploy from branch → `main` / `/ (root)`**.
Your site goes live at `https://<your-username>.github.io/<your-repo>/` within a minute or two.

Want a bare `https://<your-username>.github.io/` URL instead of a subpath? Name the repo exactly
`<your-username>.github.io` and push this same content to it.

### Custom domain (optional)
Add a `CNAME` file in this folder containing just your domain (e.g. `stefaniakonstantinidi.com`),
then point your domain's DNS at GitHub Pages (an `A` record to GitHub's IPs, or a `CNAME` record to
`<your-username>.github.io` for a subdomain). GitHub's Pages settings panel will show the exact
records once you add the domain there.

## Updating content later

- New publication → add a `.pub` block in the Publications section of `index.html`.
- Swap a photo/video → drop the new file in `assets/img/` or `assets/video/` and update the `src=`.
- New statement PDF → drop it in `assets/docs/` and link it from the Contact section.

## Still needs your input (see placeholders in index.html)

- `#linkedin-link` and `#scholar-link` / `#scholar-link-2` hrefs are `#` — paste your real URLs.
- The video caption under the "Watch" section has a note asking you to confirm what's actually in
  `What_makes_you_smile.mp4` before publishing — write one accurate sentence there.
- Research/teaching/diversity statement PDFs — drop them in `assets/docs/` and add buttons next to
  the CV download (Contact section and Publications section).
