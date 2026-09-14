document.addEventListener("DOMContentLoaded", () => {
  const input = document.querySelector("#codeowners-search");
  if (!input) return;

  const cards = [...document.querySelectorAll(".co-card")];
  const empty = document.querySelector("#codeowners-empty");

  input.addEventListener("input", () => {
    const query = input.value.trim().toLowerCase();
    let visibleCards = 0;

    cards.forEach((card) => {
      const rows = [...card.querySelectorAll(".co-rule")];
      const groupMatches = card.dataset.search.includes(query);
      let visibleRows = 0;

      rows.forEach((row) => {
        const visible = !query || groupMatches || row.dataset.search.includes(query);
        row.hidden = !visible;
        if (visible) visibleRows += 1;
      });

      card.hidden = visibleRows === 0;
      if (visibleRows) visibleCards += 1;
    });

    empty.hidden = visibleCards !== 0;
  });
});
