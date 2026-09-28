const BUTTON = {
  default: 'border-line bg-[#f3f4f6] enabled:hover:border-dim',
  primary: 'border-accent bg-accent text-white font-medium',
  danger: 'border-danger bg-danger text-white font-medium',
  ghost: 'border-line bg-transparent enabled:hover:border-dim',
  active: 'border-accent bg-[#dbeafe] text-accent',
}

export function Button({ variant = 'default', small, joined, className = '', ...props }) {
  // joined: part of a row of buttons that share their outer corners
  const shape = joined ? 'first:rounded-l last:rounded-r' : 'rounded'
  return (
    <button
      type="button"
      className={`cursor-pointer whitespace-nowrap border px-2.5 py-1 ${shape} disabled:cursor-default disabled:opacity-40 ${small ? 'text-xs' : 'text-[13px]'} ${BUTTON[variant]} ${className}`}
      {...props}
    />
  )
}

/** A label with the current value on the right, above its control. */
export function Field({ label, value, className = '', children }) {
  return (
    <div className={`mb-2 ${className}`}>
      <div className="flex justify-between text-xs text-dim">
        <span>{label}</span>
        {value != null && <output>{value}</output>}
      </div>
      {children}
    </div>
  )
}

export function Range({ value, onChange, ...props }) {
  return <input type="range" value={value} onChange={(e) => onChange(Number(e.target.value))} {...props} />
}

/** A row of buttons of which one is active. */
export function Segmented({ options, value, onChange, className = '' }) {
  return (
    <div className={`flex ${className}`}>
      {options.map((o) => (
        <Button
          key={o.value}
          small
          joined
          variant={o.value === value ? 'active' : 'default'}
          className="flex-1"
          disabled={o.disabled}
          title={o.title}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </Button>
      ))}
    </div>
  )
}

export function Swatches({ colors, onPick }) {
  return (
    <span className="flex flex-wrap items-center gap-1.5">
      {colors.map((c) => (
        <button
          key={c}
          type="button"
          title={c}
          className="h-[18px] w-[18px] cursor-pointer rounded-full border border-line"
          style={{ background: c }}
          onClick={() => onPick(c)}
        />
      ))}
    </span>
  )
}

export function SectionTitle({ children, actions }) {
  return (
    <div className="mt-4 mb-2 flex items-center justify-between font-bold">
      <span>{children}</span>
      {actions && <span className="flex gap-1.5">{actions}</span>}
    </div>
  )
}
