# Commit Standard For SMap

Tai lieu nay la chuan commit chinh thuc cho SMap. Muc tieu la giu lich su de doc, de review va de audit.

## Format

SMap nen dung Conventional Commits:

```text
type(scope): subject
```

Subject nen dai toi da 72 ky tu khi co the. Body nen wrap khoang 72-88 ky tu moi dong.

Vi du tu lich su da duoc chuan hoa:

```text
chore(project): initialize repository and early structure
feat(core): implement initial debugging UI and core logic stubs
build(ci): establish build system and CI foundation
ci(test): integrate initial tests and refine workflows
test(ci): add visual tests and enhance CI publishing
refactor(release): prepare for v1.0.0-rc.1 and cleanup assets
feat(core): refine SMap algorithm and calibration notebook
fix(docs): apply v1.0.4 hotfix and finalize documentation
build(deps): prepare for the new update and cleanup codebase
```

## Types

- `feat`: them tinh nang hoac kha nang moi.
- `fix`: sua bug hoac sua sai lech hanh vi.
- `docs`: chi thay doi tai lieu, README, notebook huong dan, hinh anh minh hoa tai lieu.
- `test`: them hoac sua test.
- `ci`: thay doi GitHub Actions hoac pipeline.
- `build`: thay doi packaging, dependency, release, setup.
- `refactor`: doi cau truc code khong doi hanh vi.
- `chore`: viec bao tri repo khong thuoc cac nhom tren.
- `perf`: toi uu hieu nang khong doi API.
- `style`: thay doi format khong doi logic.

## Scopes De Xuat

- `core`: `smap/`, thuat toan va public API.
- `docs`: `README.md`, `docs/`, tai lieu tutorial.
- `test`: `tests/`, du lieu test, vtest.
- `ci`: `.github/workflows/`.
- `deps`: `requirements.txt`, `environment.yml`, dependency setup.
- `release`: version, tag, PyPI, release artifact.
- `hdvo`: `tools/testing/hdvo/`.
- `vtest`: `tools/testing/vtest/`.
- `assets`: logo, image, media.
- `project`: khoi tao hoac cau truc repo.

## Subject Rules

- Dung imperative mood ngan gon: `add`, `fix`, `refactor`, `update` khi that su can.
- Viet bang tieng Anh de dong bo voi lich su hien tai.
- Khong ket thuc bang dau cham.
- Tranh message chung chung nhu `Update README.md`, `Add files via upload`, `Maintain unittest-passed status`.
- Neu commit gom nhieu file, subject phai noi muc dich, khong noi thao tac.

## Body Rules

Body la tuy chon, nhung nen co khi:

- Commit thay doi hanh vi thuat toan.
- Commit lien quan release/version/license.
- Commit gom nhieu thay doi lien quan cung mot muc dich.
- Commit co migration, conflict resolution, hoac tradeoff can ghi lai.

Mau body:

```text
Explain why the change is needed and what behavior it preserves.

Refs: #issue-number
```

Voi commit lon, body nen tra loi 3 cau:

- Van de hoac muc tieu la gi.
- Trang thai/hanh vi nao duoc giu lai.
- Co tradeoff hoac resolution quan trong nao can audit ve sau.

## Signing Rules

Tat ca commit moi tren branch chinh nen duoc ky bang GPG key:

```text
685C0F023DA361CC
```

Cau hinh Git de ky mac dinh:

```bash
git config user.signingkey 685C0F023DA361CC
git config commit.gpgsign true
```

Kiem tra commit moi nhat:

```bash
git log -1 --format="%h %G? %GK %GS %s"
```

Ket qua mong muon cho commit moi:

```text
<hash> G 685C0F023DA361CC Thien An L. Nguyen <thienannguyen.cv@gmail.com> <subject>
```

## Verification

Truoc khi push, chay test toi thieu:

```bash
python -m unittest discover -s tests -p "[vu]test*.py"
```

## Local Hook

Repo co the dung hook commit message trong `.githooks/commit-msg`.

Bat hook:

```bash
git config core.hooksPath .githooks
```

Hook nay chi kiem tra subject theo Conventional Commits. Chu ky GPG van duoc kiem tra bang `git log --format` nhu tren.

## CI/CD

Repo da co CI test. Lint duoc them theo 2 giai doan de tranh lam CI do dot ngot:

1. Soft gate: workflow `lint.yml` chay `pylint`, cho phep fail de lay baseline.
2. Hard gate: sau khi fix baseline, bo `continue-on-error` va bat PR phai qua lint.

Workflow commit standards kiem tra subject cac commit tren PR/push theo Conventional Commits. Neu can kiem tra GPG key tren remote, uu tien dung GitHub branch protection va Verified signature policy thay vi tu verify trong runner khong co public keyring.

Sau khi co baseline tot, co the chuyen sang hard gate.
