# GitHub publication workflow

## Prepare the repository

Check `configs/experiment.yaml` for private absolute paths, remove local outputs, and inspect the Git status before the first commit. The supplied `.gitignore` excludes common data, embedding, result, document, and checkpoint formats.

Because the working directory containing this project may already be a Git repository, either move this folder to a clean location or deliberately initialise it as a separate nested repository.

```powershell
Set-Location <repository-folder>
git init
git branch -M main
git add .
git status
git commit -m "Release reproducible MSCKG experiments"
```

Create an empty GitHub repository in the web interface, then connect and push:

```powershell
git remote add origin https://github.com/ACCOUNT/MSCKG-Virtual-Trajectory-Classification.git
git push -u origin main
```

With GitHub CLI:

```powershell
gh auth login
gh repo create MSCKG-Virtual-Trajectory-Classification --public --source . --remote origin --push
```

## Verify the public release

1. Open the repository in a signed-out browser window.
2. Confirm that no data, `.npy`, `.npz`, checkpoints, results, Word files, Excel files, or local absolute paths were committed.
3. Follow the README installation steps in a clean environment.
4. Run `python -m compileall src scripts` and `msckg --help`.
5. Create a tagged release, for example `v1.0.0`.
6. For a persistent scholarly identifier, connect the repository to Zenodo and archive the release.

## Text for the manuscript

Replace the placeholder after the public repository is available:

> Code availability. The source code used to construct the MSCKG, learn entity representations, derive trajectory features, perform classification and sensitivity analyses, and generate the reported statistics and figures is available at https://github.com/ACCOUNT/MSCKG-Virtual-Trajectory-Classification (release v1.0.0). The virtual-trajectory data are not publicly available because of data-access and privacy restrictions.

If a Zenodo DOI is created:

> Code availability. The source code is available at https://github.com/ACCOUNT/MSCKG-Virtual-Trajectory-Classification and is archived at https://doi.org/DOI. The virtual-trajectory data are not publicly available because of data-access and privacy restrictions.

The same URL can replace the blank GitHub address in the response to reviewer R2-7.
