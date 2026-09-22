export function Glossary() {
  const items = [
    {
      term: "guard",
      definition: "The scheduler upper bound. If actual lands above it, the run missed the guard.",
    },
    {
      term: "support",
      definition: "How many usable rows existed inside the chosen bucket.",
    },
    {
      term: "fallback",
      definition: "How far matching had to relax before the estimator found a usable bucket.",
    },
    {
      term: "abstain",
      definition: "No prediction was made for that target. It is not the same as task failure.",
    },
  ];

  return (
    <div className="glossary-grid">
      {items.map((item) => (
        <article key={item.term} className="glossary-card">
          <strong>{item.term}</strong>
          <p className="small-muted">{item.definition}</p>
        </article>
      ))}
    </div>
  );
}
