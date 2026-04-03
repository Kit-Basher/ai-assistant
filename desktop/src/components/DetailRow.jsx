export default function DetailRow({ badge = null, children = null, metaLines = [], title }) {
  return (
    <div className="model-row">
      <div className="model-head">
        <span>{title}</span>
        {badge}
      </div>
      {metaLines.filter(Boolean).map((line, index) => (
        <div key={index} className="meta-line">
          {line}
        </div>
      ))}
      {children}
    </div>
  );
}
