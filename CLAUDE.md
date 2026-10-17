# claude-chat-website

This repo holds web pages Claude makes during chats. It's published with GitHub Pages at <https://egodfrey-sunflower.github.io/claude-chat-website/> so the user can open the pages on their phone. The `/use-website-repo` skill describes the workflow. It lives in `skills/use-website-repo/` in this repo and is symlinked from `~/.claude/skills/use-website-repo`. Edit it here.

## ⚠️ This repo is public

Everything committed here is visible to anyone, and stays in git history even after it's deleted. **Don't commit personal details about the user.** That means:

- No names, email addresses, phone numbers, home or work addresses.
- No travel plans, dates they'll be away, or where they're staying.
- No health, financial, work or family details, and no private conversations.
- No secrets, tokens, API keys or credentials of any kind.
- No content copied from private sources (private repos, emails, documents).

Write pages for a general reader. "Weekend guide to a seaside town" is fine; "<name>'s trip there, staying at X from Friday" is not. If a page can't be useful without personal details, don't put it here. Say so, and suggest an alternative such as a private artifact or a file sent directly. If in doubt, ask before committing.

Commits use the GitHub noreply email (`…@users.noreply.github.com`). Don't change the git identity.

## Layout

Organise pages by the date they were made:

```
index.html               # home page: links to every page, newest first
YYYYMM/
  DD/
    page-name.html       # one page
    page-name/           # images, data, CSS/JS for that page
      ...
```

- Name pages in short kebab-case. If the name is already taken that day, choose a more specific one.
- Link assets relatively (`page-name/map.png`), so they work both locally and on Pages.
- Add every new page to the list in the root `index.html`, newest first, with its date.
- `.nojekyll` turns off Jekyll, so files are served exactly as committed.

## Page guidelines

- Each page is a single self-contained HTML file where possible, with inline CSS and JS. Larger assets go in its folder.
- Make it phone-first: `<meta name="viewport" content="width=device-width, initial-scale=1">`, readable font sizes, no horizontal scrolling, and support for both light and dark mode.
- Load external libraries only from well-known CDNs, pinned to an exact version.
- Put a short "Made with Claude, <date>" line and a link back to the home page (`../../index.html`) at the bottom of each page.
